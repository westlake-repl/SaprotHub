import json
import torch

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCTokenClassificationDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            model_name: name of the model
            max_length: max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)

        temp_model = ESMC.from_pretrained(model_name)
        self.tokenizer = temp_model.tokenizer
        del temp_model

        self.model_name = model_name
        self.max_length = max_length

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq']

        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        # Add -1 to match special tokens convention if needed; we will align length in loss
        label = [-1] + entry["label"][:self.max_length] + [-1]
        label = torch.tensor(label, dtype=torch.long)

        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        from data.data_transform import pad_sequences

        seqs, label_ids = tuple(zip(*batch))
        label_ids = pad_sequences(label_ids, constant_value=-1)
        labels = {"labels": label_ids}

        proteins = [ESMProtein(sequence=s) for s in seqs]
        inputs = {"inputs": {"proteins": proteins}}

        return inputs, labels


