import torchmetrics
import torch
import torch.nn.functional as F

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ESMCBaseModel


@register_model
class ESMCPairClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels

            **kwargs: other arguments for ESMCBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="pair_classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

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

        reps_1 = F.layer_norm(reps_1, reps_1.shape[-1:])
        reps_2 = F.layer_norm(reps_2, reps_2.shape[-1:])

        mask_1 = attention_mask_1.unsqueeze(-1)
        mask_2 = attention_mask_2.unsqueeze(-1)

        h1 = (reps_1 * mask_1).sum(dim=1) / mask_1.sum(dim=1).clamp(min=1)
        h2 = (reps_2 * mask_2).sum(dim=1) / mask_2.sum(dim=1).clamp(min=1)

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
        
        return head(hidden_concat)

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)