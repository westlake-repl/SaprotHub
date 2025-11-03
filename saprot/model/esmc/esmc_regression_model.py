import torchmetrics
import torch

from torch.nn.utils.rnn import pad_sequence
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

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Tokenize and pad
            token_ids_list = [self.model.embed(p) for p in proteins]
            token_ids_batch = pad_sequence(token_ids_list, batch_first=True, padding_value=self.model.padding_idx)
            
            # Forward pass to get representations
            model_output = self.model.forward(token_ids_batch, repr_layers=[self.model.num_layers])
            representations = model_output['representations'][self.model.num_layers]
            
            # Pool with masking
            mask = (token_ids_batch != self.model.padding_idx).unsqueeze(-1)
            sequence_lengths = mask.sum(dim=1)
            repr_tensor = (representations * mask).sum(dim=1) / sequence_lengths

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


