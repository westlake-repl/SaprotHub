import torchmetrics
import torch

from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCPairRegressionModel(ESMCBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        hidden_size = getattr(getattr(self.model, 'config', None), 'hidden_size', 960) * 2
        reg_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
        setattr(self.model, "reg_head", reg_head)

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs_1, inputs_2):
        proteins_1 = inputs_1.get('proteins')
        proteins_2 = inputs_2.get('proteins')

        out1 = self.model.encode(proteins_1, return_hidden_states=False)
        out2 = self.model.encode(proteins_2, return_hidden_states=False)

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
        return self.model.reg_head(hidden_concat).squeeze(-1)

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        for metric in self.metrics[stage].values():
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)

        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss


