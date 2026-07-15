import json
from typing import List, Sequence, Tuple

import lmdb
import torch
from transformers import AutoTokenizer

from ..lmdb_dataset import LMDBDataset

try:
    from saprot.data.pdb2prosst import (
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        pad_structure_input_ids,
        validate_sequence_and_structure,
    )
except ImportError:
    from data.pdb2prosst import (
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        pad_structure_input_ids,
        validate_sequence_and_structure,
    )


class ProSSTDatasetBase(LMDBDataset):
    def __init__(
        self,
        tokenizer: str,
        max_length: int = 2046,
        structure_vocab_size: int = 2048,
        structure_cache_dir: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
        )
        self.max_length = max_length
        self.structure_vocab_size = structure_vocab_size
        self.structure_cache_dir = structure_cache_dir

    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()

        self.env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.operator = self.env.begin()

    def _load_entry(self, index):
        return json.loads(self._get(index))

    def _load_sequence_and_structure(self, entry) -> Tuple[str, List[int]]:
        sequence = entry["seq"].strip().upper()
        structure_tokens = get_structure_tokens_from_entry(
            entry,
            cache_dir=self.structure_cache_dir,
            structure_vocab_size=self.structure_vocab_size,
        )
        validate_sequence_and_structure(sequence, structure_tokens)

        return sequence[: self.max_length], structure_tokens[: self.max_length]

    def _collate_prosst_inputs(
        self,
        sequences: Sequence[str],
        structure_tokens_list: Sequence[Sequence[int]],
    ):
        encoder_info = self.tokenizer.batch_encode_plus(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length + 2,
        )
        target_length = encoder_info["input_ids"].shape[1]
        encoded_structure = [
            encode_structure_tokens(tokens, self.structure_vocab_size)
            for tokens in structure_tokens_list
        ]
        ss_input_ids = pad_structure_input_ids(encoded_structure, target_length)
        encoder_info["ss_input_ids"] = torch.tensor(ss_input_ids, dtype=torch.long)

        return encoder_info
