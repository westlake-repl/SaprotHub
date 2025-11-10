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

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq']

        if not isinstance(self.tokenizer, EsmTokenizer):
            seq = " ".join(seq)

        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)
        
        # Add -1 to the start and end of the label to ignore the cls token
        label = [-1] + entry["label"][:self.max_length] + [-1]
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

        input_len = encoder_info["input_ids"].shape[-1]
        label_len = label_ids.shape[-1]

        if label_len < input_len:
            pad_width = input_len - label_len
            pad_tensor = torch.full(
                (label_ids.shape[0], pad_width),
                -1,
                dtype=label_ids.dtype,
                device=label_ids.device,
            )
            label_ids = torch.cat([label_ids, pad_tensor], dim=-1)
        elif label_len > input_len:
            label_ids = label_ids[..., :input_len]

        labels["labels"] = label_ids
 
        return inputs, labels