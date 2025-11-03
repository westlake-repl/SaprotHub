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
            if isinstance(out, torch.Tensor):
                if out.dim() == 3:
                    pooled.append(out.mean(dim=1).squeeze(0))
                elif out.dim() == 2:
                    pooled.append(out.mean(dim=0))
                elif out.dim() == 1:
                    pooled.append(out)
                else:
                    raise ValueError("Unsupported tensor shape from ESMC encode: {}".format(tuple(out.shape)))
                continue

            if hasattr(out, 'sequence_representation') and out.sequence_representation is not None:
                rep = out.sequence_representation
                pooled.append(rep if rep.dim() == 1 else rep.squeeze(0))
                continue

            hs = None
            for attr_name in ['token_representations', 'residue_representations', 'hidden_states', 'last_hidden_state']:
                if hasattr(out, attr_name) and getattr(out, attr_name) is not None:
                    hs = getattr(out, attr_name)
                    break
            if hs is None:
                raise ValueError("ESMC encode outputs lack representations compatible with pooling")
            if isinstance(hs, list):
                pooled.append(hs[0].mean(dim=0))
            elif isinstance(hs, torch.Tensor):
                if hs.dim() == 3:
                    pooled.append(hs.mean(dim=1).squeeze(0))
                elif hs.dim() == 2:
                    pooled.append(hs.mean(dim=0))
                elif hs.dim() == 1:
                    pooled.append(hs)
                else:
                    raise ValueError("Unsupported representation tensor shape: {}".format(tuple(hs.shape)))
            else:
                raise ValueError("Unsupported representation type: {}".format(type(hs)))

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


