import torch.distributed as dist
import torchmetrics
import torch
import warnings

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
        # --- DEBUG: Flag to ensure debug info is printed only once ---
        self._has_printed_debug_report = False

    def _print_debug_report(self, stage, inputs, labels, repr_before_head, outputs, loss):
        """Prints a detailed report for one training step for debugging."""
        if dist.get_rank() == 0 and not self._has_printed_debug_report:
            print("\n" + "="*40)
            print(f"DEBUG REPORT FOR: {self.__class__.__name__} (Stage: {stage})")
            print("="*40)

            # --- 1. Check Inputs & Labels ---
            print("\n--- 1. INPUTS & LABELS ---")
            fitness_labels = labels['labels'].to(outputs.device, dtype=torch.float32)
            print(f"Labels (fitness) shape: {fitness_labels.shape}")
            print(f"Labels (fitness) dtype: {fitness_labels.dtype}")
            if fitness_labels.numel() > 0:
                print(f"Labels (fitness) values (min/mean/max): {fitness_labels.min().item():.4f} / {fitness_labels.mean().item():.4f} / {fitness_labels.max().item():.4f}")
            print(f"Sample Labels: {fitness_labels[:5].cpu().numpy()}")

            # --- 2. Check Model Internals ---
            print("\n--- 2. MODEL INTERNALS ---")
            print(f"Representation before head shape: {repr_before_head.shape}")
            print(f"Representation before head dtype: {repr_before_head.dtype}")
            if repr_before_head.numel() > 0:
                print(f"Representation before head values (min/mean/max/std): {repr_before_head.min().item():.4f} / {repr_before_head.mean().item():.4f} / {repr_before_head.max().item():.4f} / {repr_before_head.std().item():.4f}")
            
            # --- 3. Check Final Outputs & Loss ---
            print("\n--- 3. FINAL OUTPUTS & LOSS ---")
            print(f"Final Logits (outputs) shape: {outputs.shape}")
            print(f"Final Logits (outputs) dtype: {outputs.dtype}")
            if outputs.numel() > 0:
                print(f"Final Logits (outputs) values (min/mean/max/std): {outputs.min().item():.4f} / {outputs.mean().item():.4f} / {outputs.max().item():.4f} / {outputs.std().item():.4f}")
            print(f"Sample Predictions: {outputs[:5].cpu().numpy()}")
            print(f"Calculated Loss: {loss.item():.6f}")

            # --- 4. CRITICAL: GRADIENT CHECK (from previous step) ---
            print("\n--- 4. GRADIENT CHECK (for trainable parameters) ---")
            # This logic needs to adapt based on the model structure (ProtBERT vs ESM-style)
            trainable_params_found = False
            try:
                # General approach: iterate through all parameters and find those requiring gradients
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        trainable_params_found = True
                        grad_info = "N/A (first step)"
                        if param.grad is not None:
                            grad_mean_abs = param.grad.abs().mean().item()
                            grad_info = f"{grad_mean_abs:.8f}"
                        else:
                            grad_info = "None (NO GRADIENT!)"
                        print(f"  - Trainable Param '{name}': Grad Mean Abs = {grad_info}")
            except Exception as e:
                print(f"  - ERROR checking gradients: {e}")
            
            if not trainable_params_found:
                print("  - WARNING: No trainable parameters found in the entire model!")

            print("\n" + "="*40)
            print("DEBUG REPORT END")
            print("="*40 + "\n")
            
            self._has_printed_debug_report = True # Set flag to true after printing
    
    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}
    
    def forward(self, inputs, structure_info=None):
        if structure_info:
            raise NotImplementedError

        repr_before_head = None
        # For ESM models
        if hasattr(self.model, "esm"):
            if self.freeze_backbone:
                repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
                x = self.model.classifier.dropout(repr)
                x = self.model.classifier.dense(x)
                repr_before_head = torch.tanh(x) # This is the input to the final layer
                x = self.model.classifier.dropout(repr_before_head)
                logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
            else:
                outputs = self.model(**inputs)
                # Assuming the structure is backbone -> classifier -> logits
                # We need to get the input to the final layer. This can be tricky.
                # A common pattern is that the logits come from a final nn.Linear layer.
                # Here we assume the direct output `outputs.logits` is what we need.
                # For a more detailed debug, one might need to hook into the model.
                logits = outputs.logits.squeeze(dim=-1)
                # Placeholder for repr_before_head in this case
                repr_before_head = torch.zeros(1) # Not easily accessible without model hooks

        # For ProtBERT
        elif hasattr(self.model, "bert"):
            repr_before_head = self.model.bert(**inputs).last_hidden_state[:, 0]
            logits = self.model.classifier(repr_before_head).squeeze(dim=-1)
        
        # Store for debugging
        self._last_repr_before_head = repr_before_head
        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        # --- DEBUG: Call the debug report function ---
        if stage == "train":
            try:
                self._print_debug_report(stage, None, labels, self._last_repr_before_head, outputs, loss)
            except Exception as e:
                if dist.get_rank() == 0:
                    warnings.warn(f"DEBUG REPORT FAILED with error: {e}")

        # Update metrics
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

            if dist.get_rank() == 0:
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