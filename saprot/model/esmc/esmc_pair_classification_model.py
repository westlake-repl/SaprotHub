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

        out1 = self.model.encode(proteins_1)
        out2 = self.model.encode(proteins_2)

        if hasattr(out1, 'sequence_representation'):
            h1 = out1.sequence_representation
            h2 = out2.sequence_representation
        elif hasattr(out1, 'hidden_states'):
            h1 = out1.hidden_states.mean(dim=1)
            h2 = out2.hidden_states.mean(dim=1)
        else:
            h1 = out1.last_hidden_state.mean(dim=1)
            h2 = out2.last_hidden_state.mean(dim=1)

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


