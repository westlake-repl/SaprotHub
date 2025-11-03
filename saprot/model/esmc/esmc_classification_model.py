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

        # Get per-sequence representations - try different ESMC API methods
        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Method 1: Try forward(proteins) which should return representations
            try:
                if isinstance(proteins, list):
                    outputs = self.model.forward(proteins)
                else:
                    outputs = self.model.forward([proteins])
                
                # Extract representations from outputs
                if isinstance(outputs, torch.Tensor):
                    # Direct tensor [B, L, D] or [B, D]
                    if outputs.dim() == 3:  # [B, L, D]
                        repr_tensor = outputs.mean(dim=1)
                    elif outputs.dim() == 2:  # [B, D]
                        repr_tensor = outputs
                    else:
                        raise ValueError("Unexpected forward output shape: {}".format(tuple(outputs.shape)))
                else:
                    # Try to extract from structured output
                    raise NotImplementedError("Need to handle structured forward output")
            except Exception as e:
                print(f"[ESMC][DEBUG] forward failed: {e}")
                raise
        
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



