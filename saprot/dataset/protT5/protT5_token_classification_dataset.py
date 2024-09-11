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
        seq = " ".join(seq)

        # Add -1 to the start and end of the label to ignore the cls token
        label = entry["label"][:self.max_length] + [-1]
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))
        # seqs = batch[0][0]
        # label_ids = batch[0][1]
        # seqs, label_ids = [(seq[0], seq[1]) for seq in batch]

        seqs = tuple(" ".join(seq) for seq in seqs)

        label_ids = pad_sequences(label_ids, constant_value=0)
        labels = {"labels": label_ids}

        encoder_info = self.tokenizer.batch_encode_plus(seqs, padding=True, tuncation=True, return_tensors='pt', max_length=self.max_length)
        inputs = {"inputs": encoder_info}
 
        return inputs, labels