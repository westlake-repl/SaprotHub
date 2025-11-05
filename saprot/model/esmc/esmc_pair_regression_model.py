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
class ESMCPairRegressionModel(ESMCBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        
        # Get hidden_size from base model (already initialized in super())
        # Try multiple methods to be robust
        hidden_size = None
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        elif hasattr(self.model, 'embed') and hasattr(self.model.embed, 'embedding_dim'):
            hidden_size = self.model.embed.embedding_dim
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            # PEFT wrapped model
            base_model = self.model.base_model.model
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            elif hasattr(base_model, 'embed') and hasattr(base_model.embed, 'embedding_dim'):
                hidden_size = base_model.embed.embedding_dim
        
        # Fallback: infer from model_name
        if hidden_size is None:
            model_name = getattr(self, 'model_name', 'esmc_300m')
            if '600m' in model_name.lower() or '600' in model_name:
                hidden_size = 1152
            else:
                hidden_size = 960
        
        hidden_size = hidden_size * 2  # Pair models need 2x hidden size
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
        if not isinstance(proteins_1, list):
            proteins_1 = [proteins_1]
        if not isinstance(proteins_2, list):
            proteins_2 = [proteins_2]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Process protein set 1
            token_ids_list_1 = [self.model.embed(p) for p in proteins_1]
            token_ids_batch_1 = pad_sequence(token_ids_list_1, batch_first=True, padding_value=self.model.padding_idx)
            model_output_1 = self.model.forward(token_ids_batch_1, repr_layers=[self.model.num_layers])
            repr_1 = model_output_1['representations'][self.model.num_layers]
            mask_1 = (token_ids_batch_1 != self.model.padding_idx).unsqueeze(-1)
            h1 = (repr_1 * mask_1).sum(dim=1) / mask_1.sum(dim=1)
            
            # Process protein set 2
            token_ids_list_2 = [self.model.embed(p) for p in proteins_2]
            token_ids_batch_2 = pad_sequence(token_ids_list_2, batch_first=True, padding_value=self.model.padding_idx)
            model_output_2 = self.model.forward(token_ids_batch_2, repr_layers=[self.model.num_layers])
            repr_2 = model_output_2['representations'][self.model.num_layers]
            mask_2 = (token_ids_batch_2 != self.model.padding_idx).unsqueeze(-1)
            h2 = (repr_2 * mask_2).sum(dim=1) / mask_2.sum(dim=1)

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


