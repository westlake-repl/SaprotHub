import json
import copy
import torch

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset


@register_dataset
class ESMCSequenceDesignDataset(LMDBDataset):
    """
    Masked token reconstruction dataset for ESMC. We use '<mask>' to mask amino acids and
    labels are 0..19 for AAs, -1 elsewhere.
    """
    def __init__(self,
                 max_length: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.aa_list = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        self.aa2id = {a:i for i,a in enumerate(self.aa_list)}

    def __len__(self):
        return int(self._get("length"))

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length]

        # mask all positions
        labels = torch.full((len(seq) + 2,), -1, dtype=torch.long)
        masked_chars = list(seq)
        for i, ch in enumerate(seq):
            labels[i+1] = self.aa2id.get(ch, -1)
            masked_chars[i] = '<mask>'

        masked_seq = "".join(masked_chars)
        return masked_seq, labels

    def collate_fn(self, batch):
        from data.data_transform import pad_sequences
        seqs, label_ids = tuple(zip(*batch))
        label_ids = pad_sequences(label_ids, -1)
        labels = {"labels": label_ids}

        try:
            from esm.sdk.api import ESMProtein
            proteins = [ESMProtein(sequence=s) for s in seqs]
        except Exception:
            proteins = seqs

        inputs = {"inputs": {"proteins": proteins}}
        return inputs, labels


