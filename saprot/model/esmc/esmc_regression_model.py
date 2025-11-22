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
        # --- DEBUG: Flag to ensure debug info is printed only once ---
        self._has_printed_debug_report = False

    def _print_debug_report(self, stage, inputs, labels, pooled_repr, outputs, loss):
        """Prints a detailed report for one training step for debugging."""
        # --- FIX: Make distributed check robust for single-GPU and multi-GPU ---
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        if should_print and not self._has_printed_debug_report:
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
            print(f"Pooled Representation shape: {pooled_repr.shape}")
            print(f"Pooled Representation dtype: {pooled_repr.dtype}")
            if pooled_repr.numel() > 0:
                print(f"Pooled Representation values (min/mean/max/std): {pooled_repr.min().item():.4f} / {pooled_repr.mean().item():.4f} / {pooled_repr.max().item():.4f} / {pooled_repr.std().item():.4f}")
            
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
            head = self._get_head()
            has_trainable_params = False
            for name, param in head.named_parameters():
                if param.requires_grad:
                    has_trainable_params = True
                    grad_info = "N/A (first step)"
                    if param.grad is not None:
                        grad_mean_abs = param.grad.abs().mean().item()
                        grad_info = f"{grad_mean_abs:.8f}"
                    else:
                        grad_info = "None (NO GRADIENT!)"
                    print(f"  - Head Param '{name}': Grad Mean Abs = {grad_info}")
            
            if not has_trainable_params:
                print("  - WARNING: No trainable parameters found in the regression head!")

            print("\n" + "="*40)
            print("DEBUG REPORT END")
            print("="*40 + "\n")
            
            self._has_printed_debug_report = True # Set flag to true after printing

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs, coords=None):
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)

        # Tokenization & Padding
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)

        # Backbone representations
        representations = self._get_representations(token_ids_batch)

        # Pooling - always needs gradients for head training
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)

        # CRITICAL FIX: Directly use modules_to_save.default if it exists
        # This ensures we use the same weight object that's being trained
        base_model = self._get_base_model()
        head = None
        
        # Fallback to _get_head() if modules_to_save.default not found
        if head is None:
            head = self._get_head()
        
        logits = head(pooled_repr).squeeze(dim=-1)

        # Store intermediate tensor for debugging
        self._last_pooled_repr = pooled_repr

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        # --- DEBUG: Call the debug report function ---
        # This happens at the beginning of the step, so gradients are from the PREVIOUS step.
        if stage == "train":
            try:
                self._print_debug_report(stage, None, labels, self._last_pooled_repr, outputs, loss)
            except Exception as e:
                # --- FIX: Make distributed check robust ---
                should_warn = (not dist.is_initialized() or dist.get_rank() == 0)
                if should_warn:
                    warnings.warn(f"DEBUG REPORT FAILED with error: {e}")

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)

        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)

            # Reset train metrics
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

            # --- FIX: Make distributed check robust ---
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