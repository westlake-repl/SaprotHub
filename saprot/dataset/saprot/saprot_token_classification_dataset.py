import json
import random
import torch

from ..data_interface import register_dataset
from transformers import AutoTokenizer, EsmTokenizer
from ..lmdb_dataset import *
from data.data_transform import pad_sequences


@register_dataset
class SaprotTokenClassificationDataset(LMDBDataset):
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        special_tokens = [
            getattr(self.tokenizer, "cls_token", None),
            getattr(self.tokenizer, "eos_token", None),
        ]
        self._special_tokens = {tok for tok in special_tokens if tok is not None}

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq']

        if not isinstance(self.tokenizer, EsmTokenizer):
            seq = " ".join(seq)

        tokens = self.tokenizer.tokenize(seq)

        # Remove tokenizer-specific special tokens so that the number of tokens
        # aligns with the residue labels.
        while tokens and tokens[0] in self._special_tokens:
            tokens.pop(0)
        while tokens and tokens[-1] in self._special_tokens:
            tokens.pop()

        tokens = tokens[:self.max_length]
        seq = " ".join(tokens)
        
        # Add -1 to the start and end of the label to ignore the cls token
        label = [-1] + entry["label"][:len(tokens)] + [-1]
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))

        label_ids = pad_sequences(label_ids, constant_value=-1)
        labels = {"labels": label_ids}

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}
 
        return inputs, labels