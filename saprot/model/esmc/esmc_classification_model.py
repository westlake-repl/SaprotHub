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

        # Build pooled representations (robust to tensor/object returns)
        pooled = []
        for idx, out in enumerate(outputs_list):
            # Tensor returns
            if isinstance(out, torch.Tensor):
                if out.dim() == 3:           # [B, L, D] (assume B==1 per single encode)
                    pooled.append(out.mean(dim=1).squeeze(0))
                elif out.dim() == 2:         # [L, D]
                    pooled.append(out.mean(dim=0))
                elif out.dim() == 1:         # [D]
                    pooled.append(out)
                else:
                    raise ValueError("Unsupported tensor shape from ESMC encode: {}".format(tuple(out.shape)))
                continue

            # Object returns with common fields
            if hasattr(out, 'sequence_representation') and out.sequence_representation is not None:
                rep = out.sequence_representation
                pooled.append(rep if rep.dim() == 1 else rep.squeeze(0))
                continue

            hs = None
            for attr_name in [
                'token_representations', 'residue_representations', 'hidden_states', 'last_hidden_state'
            ]:
                if hasattr(out, attr_name) and getattr(out, attr_name) is not None:
                    hs = getattr(out, attr_name)
                    break

            if hs is None:
                # Debug print to understand the structure
                print(f"[ESMC][DEBUG] Sample output #{idx} type: {type(out)}, dir: {dir(out)}")
                if hasattr(out, '__dict__'):
                    print(f"[ESMC][DEBUG] Sample output #{idx} __dict__: {list(out.__dict__.keys())}")
                # Try to access .representation directly (some ESMC versions use this)
                if hasattr(out, 'representation') and out.representation is not None:
                    rep = out.representation
                    pooled.append(rep if rep.dim() == 1 else rep.squeeze(0) if rep.dim() > 1 else rep)
                    continue
                raise ValueError("ESMC encode outputs lack representations compatible with pooling")

            if isinstance(hs, list):
                pooled.append(hs[0].mean(dim=0))
            elif isinstance(hs, torch.Tensor):
                if hs.dim() == 3:            # [B, L, D]
                    pooled.append(hs.mean(dim=1).squeeze(0))
                elif hs.dim() == 2:          # [L, D]
                    pooled.append(hs.mean(dim=0))
                elif hs.dim() == 1:          # [D]
                    pooled.append(hs)
                else:
                    raise ValueError("Unsupported representation tensor shape: {}".format(tuple(hs.shape)))
            else:
                raise ValueError("Unsupported representation type: {}".format(type(hs)))

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


