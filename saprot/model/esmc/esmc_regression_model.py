# file: esmc/model/esm_c_regression.py

import torchmetrics
import torch.distributed as dist
import torch
import warnings

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
        # 新增: 用于记录训练步骤的计数器
        self._train_step_counter = 0

    # 修改: 调试报告函数现在接受一个 step_count 参数
    def _print_debug_report(self, stage, labels, pooled_repr, outputs, loss, step_count):
        """在指定的训练步骤打印详细的调试报告。"""
        # 确保只在主进程打印
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        if should_print:
            print("\n" + "="*60)
            # 修改: 报告标题现在会显示当前的步骤数
            print(f"DIAGNOSTIC REPORT FOR: {self.__class__.__name__} (Train Step: {step_count})")
            print("="*60)

            # 1. 检查输入和标签
            print("\n--- [1] LABELS ---")
            fitness_labels = labels['labels'].to(outputs.device, dtype=torch.float32)
            print(f"  - Shape: {fitness_labels.shape}")
            print(f"  - Dtype: {fitness_labels.dtype}")
            if fitness_labels.numel() > 0:
                print(f"  - Stats (min/mean/max): {fitness_labels.min().item():.4f} / {fitness_labels.mean().item():.4f} / {fitness_labels.max().item():.4f}")

            # 2. 检查模型中间输出
            print("\n--- [2] MODEL INTERNALS ---")
            print(f"  - Pooled Representation Shape: {pooled_repr.shape}")
            print(f"  - Pooled Representation Dtype: {pooled_repr.dtype}")
            
            # 3. 检查最终输出和损失
            print("\n--- [3] FINAL OUTPUTS & LOSS ---")
            print(f"  - Final Logits Shape: {outputs.shape}")
            print(f"  - Final Logits Dtype: {outputs.dtype}")
            print(f"  - Calculated Loss: {loss.item():.6f}")

            # 4. 关键：梯度检查！
            print("\n--- [4] GRADIENT CHECK (FOR TRAINABLE HEAD PARAMETERS) ---")
            head = self._get_head()
            has_trainable_params = False
            for name, param in head.named_parameters():
                if param.requires_grad:
                    has_trainable_params = True
                    # 梯度是在 loss.backward() 后计算的，所以我们检查的是上一步的梯度
                    if param.grad is not None:
                        # Check for NaN/Inf in gradients
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_info = "NaN or Inf (CRITICAL: Unstable gradient!)"
                        else:
                            grad_mean_abs = param.grad.abs().mean().item()
                            grad_info = f"{grad_mean_abs:.8f}"
                            if grad_mean_abs == 0.0:
                                grad_info += " (WARNING: Gradient is exactly zero!)"
                    else:
                        # 在第一个step之后，如果梯度仍然是None，则说明梯度没有回传
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

    def forward(self, inputs, coords=None):
        proteins = self._parse_proteins_input(inputs)
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)
        representations = self._get_representations(token_ids_batch)
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)
        
        head = self._get_head()
        
        # <<< 核心修改点 >>>
        # 将输出强制转换为 float32，以保证损失计算时的数值稳定性。
        logits = head(pooled_repr).squeeze(dim=-1).float()

        # 保存中间结果以供调试报告使用
        if self.training:
            self._last_pooled_repr = pooled_repr

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        # 主要修改: 实现周期性打印逻辑
        if stage == "train":
            # 每次训练步骤，计数器加一
            self._train_step_counter += 1
            
            # 在第1步打印（用于初始检查），然后每100步打印一次
            if self._train_step_counter == 1 or self._train_step_counter % 100 == 0:
                try:
                    if hasattr(self, '_last_pooled_repr'):
                        # 调用调试报告，并传入当前步骤数
                        self._print_debug_report(stage, labels, self._last_pooled_repr, outputs, loss, self._train_step_counter)
                except Exception as e:
                    should_warn = (not dist.is_initialized() or dist.get_rank() == 0)
                    if should_warn:
                        warnings.warn(f"DIAGNOSTIC REPORT FAILED at step {self._train_step_counter} with error: {e}")

        # 更新指标
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