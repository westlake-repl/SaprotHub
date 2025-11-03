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

        # Get per-sequence representations (robust to SDK differences)
        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            if isinstance(proteins, list):
                outputs_list = [self.model.encode(p) for p in proteins]
            else:
                outputs_list = [self.model.encode(proteins)]

        # Build pooled representations
        pooled = []
        for outputs in outputs_list:
            if hasattr(outputs, 'sequence_representation') and outputs.sequence_representation is not None:
                pooled.append(outputs.sequence_representation)
                continue
            hs = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hs = outputs.hidden_states
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                hs = outputs.last_hidden_state
            if hs is None:
                raise ValueError("ESMC encode outputs lack representations compatible with pooling")
            if isinstance(hs, list):
                denom = torch.tensor([hs[0].shape[0]], device=hs[0].device, dtype=hs[0].dtype)
                pooled.append(hs[0].mean(dim=0))
            else:
                pooled.append(hs.mean(dim=0))

        repr_tensor = torch.stack(pooled, dim=0)

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


