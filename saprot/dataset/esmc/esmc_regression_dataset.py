import torch
import json

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCRegressionDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 max_length: int = 1024,
                 **kwargs):
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

        fitness = entry["fitness"]
        return seq, float(fitness)

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, labels = tuple(zip(*batch))

        labels = {"labels": torch.tensor(labels, dtype=torch.float32)}

        proteins = [ESMProtein(sequence=s) for s in seqs]
        inputs = {"inputs": {"proteins": proteins}}

        return inputs, labels
