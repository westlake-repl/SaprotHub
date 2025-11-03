import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from torch.nn import Linear, ReLU
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCPairClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()

        hidden_size = getattr(getattr(self.model, 'config', None), 'hidden_size', 960) * 2
        classifier = torch.nn.Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, self.num_labels)
        )
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs_1, inputs_2):
        if not (isinstance(inputs_1, dict) and isinstance(inputs_2, dict)):
            raise ValueError("ESMCPairClassificationModel expects two dict inputs with key 'proteins'")

        proteins_1 = inputs_1.get('proteins')
        proteins_2 = inputs_2.get('proteins')

        def pool(repr):
            if isinstance(repr, torch.Tensor):
                if repr.dim() == 2:      # [L, D]
                    return repr.mean(dim=0)
                elif repr.dim() == 1:    # [D]
                    return repr
                else:
                    raise ValueError("Unexpected embed output shape: {}".format(tuple(repr.shape)))
            else:
                raise ValueError("ESMC embed returned non-tensor: {}".format(type(repr)))

        if isinstance(proteins_1, list):
            h1 = torch.stack([pool(self.model.embed(p)) for p in proteins_1], dim=0)
        else:
            h1 = pool(self.model.embed(proteins_1)).unsqueeze(0)
        if isinstance(proteins_2, list):
            h2 = torch.stack([pool(self.model.embed(p)) for p in proteins_2], dim=0)
        else:
            h2 = pool(self.model.embed(proteins_2)).unsqueeze(0)

        hidden_concat = torch.cat([h1, h2], dim=-1)
        return self.model.classifier(hidden_concat)

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


