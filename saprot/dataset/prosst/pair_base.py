from typing import List, Tuple

from .base import ProSSTDatasetBase


class ProSSTPairDatasetBase(ProSSTDatasetBase):
    def _load_pair_entry(
        self,
        entry,
        pair_index: int,
    ) -> Tuple[str, List[int]]:
        pair_entry = {
            "seq": entry[f"seq_{pair_index}"],
            "structure_tokens": entry[f"structure_tokens_{pair_index}"],
            "structure_vocab_size": entry.get("structure_vocab_size"),
        }
        return self._load_sequence_and_structure(pair_entry)

    def _collate_pair_inputs(self, batch):
        pair_inputs = [sample[:4] for sample in batch]
        sequences_1, structure_tokens_1, sequences_2, structure_tokens_2 = zip(
            *pair_inputs
        )
        return {
            "inputs_1": self._collate_prosst_inputs(
                sequences_1,
                structure_tokens_1,
            ),
            "inputs_2": self._collate_prosst_inputs(
                sequences_2,
                structure_tokens_2,
            ),
        }
