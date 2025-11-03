import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCTokenClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="token_classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCTokenClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Tokenize and pad
            token_ids_list = [self.model.embed(p) for p in proteins]
            token_ids_batch = pad_sequence(token_ids_list, batch_first=True, padding_value=self.model.padding_idx)
            
            # Forward pass to get representations
            model_output = self.model.forward(token_ids_batch, repr_layers=[self.model.num_layers])
            batch_repr = model_output['representations'][self.model.num_layers]  # [B, L, D]
            attn = (token_ids_batch != self.model.padding_idx)

        # position-wise classification
        x = self.model.classifier[0](batch_repr)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        logits = self.model.classifier[3](x)  # [B, L, C]

        return {"logits": logits, "attention_mask": attn}

    def loss_func(self, stage, outputs, labels):
        logits = outputs['logits']  # [B, L, C]
        # labels: [B, L+2] or [B, L] with -1 for ignored
        gold = labels['labels'].to(logits.device)

        # align shapes (truncate/pad if needed)
        min_len = min(gold.shape[1], logits.shape[1])
        gold = gold[:, :min_len]
        logits = logits[:, :min_len]

        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=-1)

        for metric in self.metrics[stage].values():
            metric.update(logits.detach().reshape(-1, logits.size(-1)), gold.reshape(-1))

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss


