import json
import torch
import torchmetrics

from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCMutationModel(ESMCBaseModel):
    """
    Zero-shot mutational effect prediction using ESMC per-residue logits.
    """
    def __init__(self,
                 log_clinvar: bool = False,
                 log_dir: str = None,
                 **kwargs):
        self.log_clinvar = log_clinvar
        self.log_dir = log_dir
        if log_clinvar:
            self.mut_info_list = []
        super().__init__(task="lm", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_spearman": torchmetrics.SpearmanCorrCoef()}

    @staticmethod
    def _seq_to_protein(seq: str):
        return ESMProtein(sequence=seq)

    @staticmethod
    def _aa_index():
        # Standard 20 AAs
        return ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

    def _position_logits(self, seqs: list):
        proteins = [self._seq_to_protein(s) for s in seqs]
        # Request sequence logits
        logits_cfg = LogitsConfig(sequence=True)
        outputs = self.model.logits(proteins, logits_config=logits_cfg)
        # Expect outputs.sequence_logits: List[Tensor[L, 20]] or Tensor[B, L, 20]
        if hasattr(outputs, 'sequence_logits'):
            seq_logits = outputs.sequence_logits
            if isinstance(seq_logits, list):
                # pad to tensor
                max_len = max(t.shape[0] for t in seq_logits)
                out = torch.full((len(seq_logits), max_len, seq_logits[0].shape[-1]), float('nan'), device=seq_logits[0].device)
                for i, t in enumerate(seq_logits):
                    out[i, :t.shape[0]] = t
                return out
            return seq_logits
        # fallback
        raise RuntimeError("ESMC logits API: sequence_logits not found. Please verify ESMC SDK.")

    def forward(self, wild_type, seqs, mut_info, structure_content, structure_type, plddt):
        device = self.device
        # Compute logits for original (masked) and optionally variant positions
        aa_list = self._aa_index()

        # Build masked sequences per example according to mut_info
        masked_seqs = []
        pos_list = []
        ori_list = []
        mut_list = []
        for seq, info in zip(seqs, mut_info):
            # only single/combined substitutions like A123B:C124D
            # we will apply mask token at each target position and aggregate
            tokens = list(seq)
            total = []
            for single in info.split(":"):
                ori_aa, pos, new_aa = single[0], int(single[1:-1]), single[-1]
                total.append((ori_aa, pos, new_aa))
                tokens[pos-1] = "<mask>"
            masked_seqs.append("".join(tokens))
            pos_list.append([p for _, p, _ in total])
            ori_list.append([a for a, _, _ in total])
            mut_list.append([m for _, _, m in total])

        # Get per-position logits on masked sequences
        logits = self._position_logits(masked_seqs)  # [B, L, 20]
        probs = torch.softmax(logits, dim=-1)

        preds = []
        aa2id = {aa: i for i, aa in enumerate(aa_list)}
        for i in range(len(masked_seqs)):
            pred = 0.0
            for pos, ori_aa, mut_aa in zip(pos_list[i], ori_list[i], mut_list[i]):
                p_ori = probs[i, pos-1, aa2id[ori_aa]]
                p_mut = probs[i, pos-1, aa2id[mut_aa]]
                pred += torch.log(p_mut / p_ori)
            preds.append(pred)

        preds = torch.stack(preds).to(device)

        if self.log_clinvar:
            self.mut_info_list.append((mut_info, -preds.detach().cpu()))

        return preds

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels']
        for metric in self.metrics[stage].values():
            metric.update(outputs.detach().float(), fitness.float())

    def on_test_epoch_end(self):
        spearman = self.test_spearman.compute()
        self.reset_metrics("test")
        self.log("spearman", spearman)

        if self.log_clinvar and self.log_dir is not None:
            # save per-rank logs
            name = getattr(self.trainer.datamodule, 'test_lmdb', 'unknown')
            name = name.split('/')[-1]
            import os
            os.makedirs(self.log_dir, exist_ok=True)
            path = f"{self.log_dir}/{name}.csv"
            with open(path, "w") as w:
                w.write("mutations,evol_indices\n")
                for mut, pred in self.mut_info_list:
                    for m, p in zip(mut, pred):
                        w.write(f"{m},{p.item()}\n")
            self.mut_info_list = []


