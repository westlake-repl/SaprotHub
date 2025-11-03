import torch
import torchmetrics
from torch.nn.functional import cross_entropy

from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCIFModel(ESMCBaseModel):
    """Inverse folding style masked-token reconstruction using ESMC logits API."""
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}

    @staticmethod
    def _seq_to_protein(seq: str) -> ESMProtein:
        return ESMProtein(seq=seq)

    @staticmethod
    def _aa_index():
        return ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

    def forward(self, inputs, coords=None):
        # inputs['inputs'] is expected if coming from LMDBDataset; here we allow direct dict with 'proteins'
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCIFModel.forward expects inputs['proteins'] (list of ESMProtein)")

        logits_cfg = LogitsConfig(sequence=True)
        outputs = self.model.logits(proteins, logits_config=logits_cfg)

        if hasattr(outputs, 'sequence_logits'):
            seq_logits = outputs.sequence_logits  # list of [L, 20] or Tensor[B, L, 20]
            if isinstance(seq_logits, list):
                max_len = max(t.shape[0] for t in seq_logits)
                device = seq_logits[0].device
                out = torch.full((len(seq_logits), max_len, seq_logits[0].shape[-1]), float('nan'), device=device)
                for i, t in enumerate(seq_logits):
                    out[i, :t.shape[0]] = t
                logits = out
            else:
                logits = seq_logits
        else:
            raise RuntimeError("ESMC logits API: sequence_logits not found.")

        return {"logits": logits}

    def loss_func(self, stage, outputs, labels):
        logits = outputs['logits']  # [B, L, 20]
        gold = labels['labels'].to(logits.device)  # [B, L] with -1 ignored, where values are 0..19

        min_len = min(gold.shape[1], logits.shape[1])
        gold = gold[:, :min_len]
        logits = logits[:, :min_len]

        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=-1)
        getattr(self, f"{stage}_acc").update(logits.detach().reshape(-1, logits.size(-1)), gold.reshape(-1))

        if stage == 'train':
            log_dict = self.get_log_dict('train')
            log_dict['train_loss'] = loss
            self.log_info(log_dict)
            self.reset_metrics('train')

        return loss


