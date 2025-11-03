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

        # Get per-sequence representations using ESMC's embed method
        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            if isinstance(proteins, list):
                repr_list = [self.model.embed(p) for p in proteins]
            else:
                repr_list = [self.model.embed(proteins)]
        
        # Embed returns tensor directly: each element is [L, D] or [D]
        pooled = []
        for repr in repr_list:
            if isinstance(repr, torch.Tensor):
                if repr.dim() == 2:      # [L, D]
                    pooled.append(repr.mean(dim=0))
                elif repr.dim() == 1:    # [D]
                    pooled.append(repr)
                else:
                    raise ValueError("Unexpected embed output shape: {}".format(tuple(repr.shape)))
            else:
                raise ValueError("ESMC embed returned non-tensor: {}".format(type(repr)))
        
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



