# file: esmc/model/esm_c_regression.py

import torchmetrics
import torch.distributed as dist
import torch
import warnings

# 用于梯度裁剪
from torch.nn.utils import clip_grad_norm_

from torch.nn.utils.rnn import pad_sequence
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCRegressionModel(ESMCBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for ESMCBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
        self._train_step_counter = 0

    # =========================================================================
    # <<< 核心修正点 1: 精准定位并裁剪 LoRA 框架下的可训练参数 >>>
    # =========================================================================
    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """
        在优化器更新权重之前被调用。
        这是手动实现梯度裁剪的最佳位置，可以防止梯度爆炸。
        """
        grad_clip_val = 1.0  # 梯度裁剪阈值，1.0 是一个非常安全和常用的值
        
        head = self._get_head()

        # 根据日志和 Base 模型的 LoRA 设置，我们知道真正被训练的参数
        # 在 head.modules_to_save.default 中。我们必须对这部分参数进行裁剪。
        params_to_clip = None
        if hasattr(head, 'modules_to_save') and hasattr(head.modules_to_save, 'default'):
            # 精确找到 peft 库暴露出来的可训练参数
            params_to_clip = head.modules_to_save.default.parameters()
            # 增加一个日志，确认我们找到了正确的参数
            if self._train_step_counter == 1:
                 print("\n[INFO] LoRA detected. Gradient clipping will be applied to 'head.modules_to_save.default' parameters.\n")
        else:
            # 如果没有使用 LoRA (例如 lora_kwargs=None)，则回退到裁剪整个 head
            params_to_clip = head.parameters()
            if self._train_step_counter == 1:
                 print("\n[INFO] No LoRA detected. Applying gradient clipping to all 'head' parameters.\n")

        # 对找到的参数执行梯度裁剪
        clip_grad_norm_(params_to_clip, max_norm=grad_clip_val)
    # =========================================================================

    def _print_debug_report(self, stage, labels, pooled_repr, outputs, loss, step_count):
        """在指定的训练步骤打印详细的调试报告。"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        if should_print:
            print("\n" + "="*60)
            print(f"DIAGNOSTIC REPORT FOR: {self.__class__.__name__} (Train Step: {step_count})")
            print("="*60)

            print("\n--- [1] LABELS ---")
            fitness_labels = labels['labels'].to(outputs.device, dtype=torch.float32)
            print(f"  - Shape: {fitness_labels.shape}")
            print(f"  - Dtype: {fitness_labels.dtype}")
            if fitness_labels.numel() > 0:
                print(f"  - Stats (min/mean/max): {fitness_labels.min().item():.4f} / {fitness_labels.mean().item():.4f} / {fitness_labels.max().item():.4f}")

            print("\n--- [2] MODEL INTERNALS ---")
            print(f"  - Pooled Representation Shape: {pooled_repr.shape}")
            print(f"  - Pooled Representation Dtype: {pooled_repr.dtype}")
            if pooled_repr.numel() > 0:
                print(f"  - Pooled Repr Stats (min/mean/max): {pooled_repr.min().item():.4f} / {pooled_repr.mean().item():.4f} / {pooled_repr.max().item():.4f}")

            print("\n--- [3] FINAL OUTPUTS & LOSS ---")
            print(f"  - Final Logits Shape: {outputs.shape}")
            print(f"  - Final Logits Dtype: {outputs.dtype}")
            print(f"  - Calculated Loss: {loss.item():.6f}")

            print("\n--- [4] GRADIENT CHECK (FOR TRAINABLE HEAD PARAMETERS) ---")
            head = self._get_head()
            has_trainable_params = False
            # 我们现在检查所有参数，以确认梯度流向
            for name, param in head.named_parameters():
                if param.requires_grad:
                    has_trainable_params = True
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_info = "NaN or Inf (CRITICAL: Unstable gradient!)"
                        else:
                            grad_mean_abs = param.grad.abs().mean().item()
                            grad_info = f"{grad_mean_abs:.8f}"
                            if grad_mean_abs == 0.0:
                                grad_info += " (WARNING: Gradient is exactly zero!)"
                    else:
                        grad_info = "None (If this is not step 1, it means NO GRADIENT is flowing!)"
                    print(f"  - Head Param '{name}': Grad Mean Abs = {grad_info}")
            
            if not has_trainable_params:
                print("  - WARNING: No trainable parameters found in the regression head!")

            print("\n" + "="*60)
            print(f"DIAGNOSTIC REPORT END (Step: {step_count})")
            print("="*60 + "\n")

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    # =========================================================================
    # <<< 核心修正点 2: 稳定回归头的输入 >>>
    # =========================================================================
    def forward(self, inputs, coords=None):
        proteins = self._parse_proteins_input(inputs)
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)
        representations = self._get_representations(token_ids_batch)
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)
        
        # 对池化后的表征进行 Layer Normalization
        # 这一步是防止梯度爆炸的第一道防线，确保回归头的输入总是处于一个稳定的数值范围。
        if not hasattr(self, 'pool_norm'):
            # 动态创建 LayerNorm 层，并确保它在正确的设备上
            self.pool_norm = torch.nn.LayerNorm(pooled_repr.size(-1)).to(pooled_repr.device)
        
        pooled_repr_norm = self.pool_norm(pooled_repr)

        head = self._get_head()
        
        # 使用标准化后的表征输入回归头
        logits = head(pooled_repr_norm).squeeze(dim=-1).float()

        # 在训练期间，保存未经标准化的 pooled_repr 以便在调试报告中查看其原始范围
        if self.training:
            self._last_pooled_repr = pooled_repr

        return logits
    # =========================================================================

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        if stage == "train":
            self._train_step_counter += 1
            
            # 在第1步和之后每100步打印一次诊断报告
            if self._train_step_counter == 1 or self._train_step_counter % 100 == 0:
                try:
                    if hasattr(self, '_last_pooled_repr'):
                        self._print_debug_report(stage, labels, self._last_pooled_repr, outputs, loss, self._train_step_counter)
                except Exception as e:
                    should_warn = (not dist.is_initialized() or dist.get_rank() == 0)
                    if should_warn:
                        warnings.warn(f"DIAGNOSTIC REPORT FAILED at step {self._train_step_counter} with error: {e}")

        for metric in self.metrics[stage].values():
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)

        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors
            
            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))
            
            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            should_write = (not dist.is_initialized() or dist.get_rank() == 0)
            if should_write:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        w.write(f"{pred.item()}\t{target.item()}\n")
        
        log_dict = self.get_log_dict("test")
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        self.plot_valid_metrics_curve(log_dict)