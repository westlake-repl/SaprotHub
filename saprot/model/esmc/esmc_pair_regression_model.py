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


