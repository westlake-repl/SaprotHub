import json
import numpy as np
import torch

from ..lmdb_dataset import LMDBDataset
from ..data_interface import register_dataset

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCAnnotationDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 bias_feature: bool = False,
                 max_length: int = 1024,
                 mask_struc_ratio: float = None,
                 plddt_threshold: float = None,
                 **kwargs):
        super().__init__(**kwargs)

        # tokenizer from temporary model
        temp_model = ESMC.from_pretrained(model_name)
        self.tokenizer = temp_model.tokenizer
        del temp_model

        self.model_name = model_name
        self.bias_feature = bias_feature
        self.max_length = max_length
        self.mask_struc_ratio = mask_struc_ratio
        self.plddt_threshold = plddt_threshold

    def __getitem__(self, index):
        data = json.loads(self._get(index))
        seq = data['seq']

        # truncate
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        # structure masking not supported for ESMC tokenizer; keep as plain sequence
        coords = data.get('coords', None)
        if self.bias_feature and coords is not None:
            coords = coords[:self.max_length]
        else:
            coords = None

        label = data['label']
        if isinstance(label, str):
            label = json.loads(label)

        return seq, label, coords

    def __len__(self):
        return int(self._get('length'))

    def collate_fn(self, batch):
        seqs, labels, coords = zip(*batch)

        proteins = [ESMProtein(seq=s) for s in seqs]
        inputs = {"inputs": {"proteins": proteins}}
        if self.bias_feature and coords[0] is not None:
            inputs['structure_info'] = (coords,)

        labels = {"labels": torch.tensor(labels, dtype=torch.long)}

        return inputs, labels


