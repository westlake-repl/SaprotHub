import torch
import json
import random

from ..data_interface import register_dataset
from transformers import T5Tokenizer
from ..lmdb_dataset import *
import re

# 1. 函数名
# 2. Tokenizer
# 3. T5Tokenizer requires the SentencePiece library but it was not found in your environment.:pip install sentencepiece


@register_dataset
class ProtT5ClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 use_bias_feature: bool = False,
                 max_length: int = 1024,
                 preset_label: int = None,
                 mask_struc_ratio: float = None,
                 mask_seed: int = 20000812,
                 plddt_threshold: float = None,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            use_bias_feature: If True, structure information will be used
            max_length: Max length of sequence
            preset_label: If not None, all labels will be set to this value
            mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
            mask_seed: Seed for mask_struc_ratio
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        # self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self.mask_struc_ratio = mask_struc_ratio
        self.mask_seed = mask_seed
        self.plddt_threshold = plddt_threshold

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][::2]
        seq = " ".join(seq)
        
        label = entry["label"] if self.preset_label is None else self.preset_label

        return seq, label
    
    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, labels = tuple(zip(*batch))
        labels = torch.tensor(labels)
        labels = {"labels": labels}
        
        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
        inputs = {"inputs": encoder_info}
        
        return inputs, labels