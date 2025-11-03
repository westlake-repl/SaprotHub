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

        # Encode per-protein when list is provided
        if isinstance(proteins, list):
            outputs_list = [self.model.encode(p) for p in proteins]
        else:
            outputs_list = [self.model.encode(proteins)]

        pooled = []
        for out in outputs_list:
            if hasattr(out, 'sequence_representation') and out.sequence_representation is not None:
                pooled.append(out.sequence_representation)
                continue
            hs = getattr(out, 'hidden_states', None)
            if hs is None:
                hs = getattr(out, 'last_hidden_state', None)
            if isinstance(hs, list):
                pooled.append(hs[0].mean(dim=0))
            else:
                pooled.append(hs.mean(dim=0))

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


