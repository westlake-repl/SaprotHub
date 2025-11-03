import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        # Expect inputs to be a dict containing key 'proteins': List[ESMProtein]
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        # Get per-sequence representations
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model.encode(proteins, return_hidden_states=False)

        # Prefer sequence_representation if provided by SDK; otherwise mean over hidden states
        if hasattr(outputs, 'sequence_representation'):
            repr_tensor = outputs.sequence_representation  # [B, D]
        elif hasattr(outputs, 'hidden_states'):
            repr_tensor = outputs.hidden_states.mean(dim=1)  # [B, L, D] -> [B, D]
        else:
            # Fallback: try attribute commonly used
            repr_tensor = outputs.last_hidden_state.mean(dim=1)

        # Classification head
        x = self.model.classifier[0](repr_tensor)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        logits = self.model.classifier[3](x)

        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss


