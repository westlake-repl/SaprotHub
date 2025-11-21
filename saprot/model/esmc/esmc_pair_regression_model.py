import torchmetrics
import torch
import torch.nn.functional as F

from ..model_interface import register_model
from .base import ESMCBaseModel


@register_model
class ESMCPairRegressionModel(ESMCBaseModel):
    def __init__(self, target_mean: float = 0.0, target_std: float = 1.0, **kwargs):
        """
        Args:
            **kwargs: other arguments for ESMCBaseModel
        """
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        self._last_logits_norm = None
        super().__init__(task="pair_regression", **kwargs)

    def initialize_metrics(self, stage):
        return {
            f"{stage}_loss": torchmetrics.MeanSquaredError(),
            f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
            f"{stage}_R2": torchmetrics.R2Score(),
            f"{stage}_pearson": torchmetrics.PearsonCorrCoef(),
        }

    def forward(self, inputs_1, inputs_2):
        proteins_1 = self._parse_proteins_input(inputs_1)
        proteins_2 = self._parse_proteins_input(inputs_2)

        token_ids_1, attention_mask_1, tokenizer = self._tokenize_sequences(proteins_1)
        token_ids_2, attention_mask_2, _ = self._tokenize_sequences(proteins_2)

        base_model = self._get_base_model()
        num_layers = getattr(base_model, "num_layers", None)

        if num_layers is not None:
            reps_1 = self._get_representations(token_ids_1, repr_layers=[num_layers])
            reps_2 = self._get_representations(token_ids_2, repr_layers=[num_layers])
        else:
            reps_1 = self._get_representations(token_ids_1)
            reps_2 = self._get_representations(token_ids_2)

        # Use mean pooling like ESMC classification does
        mask_1 = attention_mask_1.unsqueeze(-1)
        mask_2 = attention_mask_2.unsqueeze(-1)

        h1 = (reps_1 * mask_1).sum(dim=1) / mask_1.sum(dim=1).clamp(min=1)
        h2 = (reps_2 * mask_2).sum(dim=1) / mask_2.sum(dim=1).clamp(min=1)

        # Normalize pooled representations for numerical stability
        h1 = self._normalize_pooled_repr(h1)
        h2 = self._normalize_pooled_repr(h2)

        hidden_concat = torch.cat([h1, h2], dim=-1)
        
        # CRITICAL FIX: Directly use modules_to_save.default if it exists
        # This ensures we use the same weight object that's being trained
        head = None
        
        # First, try to get modules_to_save.default directly
        if hasattr(base_model, 'reg_head') and hasattr(base_model.reg_head, 'modules_to_save'):
            if hasattr(base_model.reg_head.modules_to_save, 'default'):
                head = base_model.reg_head.modules_to_save.default
        elif hasattr(base_model, 'classifier') and hasattr(base_model.classifier, 'modules_to_save'):
            if hasattr(base_model.classifier.modules_to_save, 'default'):
                head = base_model.classifier.modules_to_save.default
        elif hasattr(base_model, 'head') and hasattr(base_model.head, 'modules_to_save'):
            if hasattr(base_model.head.modules_to_save, 'default'):
                head = base_model.head.modules_to_save.default
        
        # Fallback to _get_head() if modules_to_save.default not found
        if head is None:
            head = self._get_head()
        
        logits_norm = head(hidden_concat).squeeze(-1)
        self._last_logits_norm = logits_norm
        logits = logits_norm * self.target_std + self.target_mean
        
        return logits

    def loss_func(self, stage, logits, labels):
        fitness = labels['labels'].to(logits)
        fitness_norm = (fitness - self.target_mean) / self.target_std
        logits_norm = self._last_logits_norm if self._last_logits_norm is not None else (logits - self.target_mean) / self.target_std
        loss = torch.nn.functional.mse_loss(logits_norm, fitness_norm)

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(logits.detach().float(), fitness.float())

        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")

        self.plot_valid_metrics_curve(log_dict)