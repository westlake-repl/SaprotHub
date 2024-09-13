import json
import random
import torch

from ..data_interface import register_dataset
from transformers import T5Tokenizer
from ..lmdb_dataset import *
from data.data_transform import pad_sequences
import re


@register_dataset
class ProtT5TokenClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][::2]

        # Add 0 to the end of the label to ignore the cls token
        if len(entry["label"]) >= self.max_length:
            label = entry["label"][:self.max_length - 1] + [0]
        else:
            label = entry["label"][:self.max_length] + [0]  
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))
        # Add a space between each amino acid
        seqs = tuple(" ".join(seq) for seq in seqs)
        # Pad the label_ids with 0
        label_ids = pad_sequences(label_ids, constant_value=0)
        labels = {"labels": label_ids}
        
        # Encode the sequences
        encoder_info = self.tokenizer.batch_encode_plus(seqs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        inputs = {"inputs": encoder_info}

        return inputs, labels