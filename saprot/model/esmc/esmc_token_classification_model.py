import torchmetrics
import torch

from torch.nn.functional import cross_entropy
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

        # get token-level representations using embed method
        if isinstance(proteins, list):
            repr_list = [self.model.embed(p) for p in proteins]
        else:
            repr_list = [self.model.embed(proteins)]
        
        # collect per-seq token representations (ensure [L, D])
        per_seq = []
        for repr in repr_list:
            if isinstance(repr, torch.Tensor):
                if repr.dim() == 3:         # [B, L, D], assume B==1
                    per_seq.append(repr.squeeze(0))
                elif repr.dim() == 2:       # [L, D]
                    per_seq.append(repr)
                else:
                    raise ValueError("Unexpected embed output shape: {}".format(tuple(repr.shape)))
            else:
                raise ValueError("ESMC embed returned non-tensor: {}".format(type(repr)))

        max_len = max(t.shape[0] for t in per_seq)
        hidden_size = per_seq[0].shape[-1]
        device = per_seq[0].device
        batch_repr = torch.zeros(len(per_seq), max_len, hidden_size, device=device)
        attn = torch.zeros(len(per_seq), max_len, dtype=torch.bool, device=device)
        for i, t in enumerate(per_seq):
            L = t.shape[0]
            batch_repr[i, :L] = t
            attn[i, :L] = True

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


