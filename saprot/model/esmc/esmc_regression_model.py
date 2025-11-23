import torchmetrics
import torch.distributed as dist
import torch
import torch.nn.functional as F

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

    def forward(self, inputs, coords=None):
        proteins = self._parse_proteins_input(inputs)
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)
        representations = self._get_representations(token_ids_batch)
        representations = F.layer_norm(representations, representations.shape[-1:])
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)
        
        head = self._get_head()
        
        logits = head(pooled_repr).squeeze(dim=-1)
        # NOTE: This sigmoid is the most likely cause of the problem.
        # It forces all predictions to be between 0 and 1.
        logits = torch.sigmoid(logits)

        return logits

    def loss_func(self, stage, outputs, labels):
        if stage == "train":
            self._train_step_counter += 1

        # <<< CALLING THE DIAGNOSTIC FUNCTION HERE >>>
        self._print_forward_diagnostics("ESMCRegressionModel", stage, outputs, labels)

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