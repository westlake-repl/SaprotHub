import torchmetrics
import torch

from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCRegressionModel(ESMCBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCRegressionModel.forward expects inputs['proteins'] (list of ESMProtein)")

        # Use embed method to get representations
        if isinstance(proteins, list):
            repr_list = [self.model.embed(p) for p in proteins]
        else:
            repr_list = [self.model.embed(proteins)]
        
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

        x = self.model.classifier[0](repr_tensor)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        logits = self.model.classifier[3](x).squeeze(dim=-1)
        return logits

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


