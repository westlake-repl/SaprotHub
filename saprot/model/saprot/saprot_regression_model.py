import torch.distributed as dist
import torchmetrics
import torch

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
        self._train_step_counter = 0

    # ======================= DIAGNOSTIC FUNCTION =======================
    def _print_forward_diagnostics(self, model_name, stage, outputs, labels):
        """Prints a detailed report of inputs and outputs for debugging."""
        # Run only on rank 0 to avoid log spam
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        # Print only on the first step and then periodically
        if self._train_step_counter > 2 and self._train_step_counter % 100 != 0:
            return

        print("\n" + "="*30 + f" DIAGNOSTIC REPORT: {model_name} " + "="*30)
        print(f"Stage: {stage} | Global Step: {self._train_step_counter}")
        print("-" * (62 + len(model_name)))

        # 1. Analyze Labels
        fitness_labels = labels['labels'].to(outputs.device, dtype=torch.float32)
        print("\n--- [1] LABELS (Ground Truth) ---")
        print(f"  - Shape: {fitness_labels.shape}")
        print(f"  - Dtype: {fitness_labels.dtype}")
        if fitness_labels.numel() > 0:
            print(f"  - Stats (min/mean/max/std): "
                  f"{fitness_labels.min().item():.4f} / "
                  f"{fitness_labels.mean().item():.4f} / "
                  f"{fitness_labels.max().item():.4f} / "
                  f"{fitness_labels.std().item():.4f}")
            print(f"  - First 5 values: {fitness_labels.flatten()[:5].tolist()}")

        # 2. Analyze Model Outputs
        print("\n--- [2] MODEL OUTPUTS (Predictions) ---")
        print(f"  - Shape: {outputs.shape}")
        print(f"  - Dtype: {outputs.dtype}")
        if outputs.numel() > 0:
            print(f"  - Stats (min/mean/max/std): "
                  f"{outputs.min().item():.4f} / "
                  f"{outputs.mean().item():.4f} / "
                  f"{outputs.max().item():.4f} / "
                  f"{outputs.std().item():.4f}")
            print(f"  - First 5 values: {outputs.flatten()[:5].tolist()}")
        
        print("\n" + "="* (62 + len(model_name)) + "\n")
    # ====================================================================

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}
    
    def forward(self, inputs, structure_info=None):
        if structure_info:
            raise NotImplementedError

        if hasattr(self.model, "esm"):
            if self.freeze_backbone:
                repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
                x = self.model.classifier.dropout(repr)
                x = self.model.classifier.dense(x)
                x = torch.tanh(x)
                x = self.model.classifier.dropout(x)
                logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
            else:
                logits = self.model(**inputs).logits.squeeze(dim=-1)

        elif hasattr(self.model, "bert"):
            repr = self.model.bert(**inputs).last_hidden_state[:, 0]
            logits = self.model.classifier(repr).squeeze(dim=-1)

        return logits

    def loss_func(self, stage, outputs, labels):
        if stage == "train":
            self._train_step_counter += 1
        
        # <<< CALLING THE DIAGNOSTIC FUNCTION HERE >>>
        self._print_forward_diagnostics("SaprotRegressionModel", stage, outputs, labels)

        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

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

            if not dist.is_initialized() or dist.get_rank() == 0:
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