import torchmetrics
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

from saprot.model.model_interface import register_model
from saprot.model.esmc.base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for ESMCBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)

        # Tokenization & Padding
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)

        # Forward through backbone and get representations
        representations = self._get_representations(token_ids_batch)

        # Normalize representations
        representations = F.layer_norm(representations, representations.shape[-1:])
        
        # Pooling - always needs gradients for head training
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)

        # CRITICAL FIX: Directly use modules_to_save.default if it exists
        # This ensures we use the same weight object that's being trained
        base_model = self._get_base_model()
        head = None
        
        # First, try to get modules_to_save.default directly
        if hasattr(base_model, 'classifier') and hasattr(base_model.classifier, 'modules_to_save'):
            if hasattr(base_model.classifier.modules_to_save, 'default'):
                head = base_model.classifier.modules_to_save.default
        elif hasattr(base_model, 'head') and hasattr(base_model.head, 'modules_to_save'):
            if hasattr(base_model.head.modules_to_save, 'default'):
                head = base_model.head.modules_to_save.default
        
        # Fallback to _get_head() if modules_to_save.default not found
        if head is None:
            head = self._get_head()
        
        logits = head(pooled_repr)

        return logits
    
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
