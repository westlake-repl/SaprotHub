import torch

from ..data_interface import register_dataset
from .base import ProSSTDatasetBase


@register_dataset
class ProSSTMutationDataset(ProSSTDatasetBase):
    """Utility dataset for batched ProSST zero-shot mutation scoring."""

    def __getitem__(self, index):
        entry = self._load_entry(index)
        sequence, structure_tokens = self._load_sequence_and_structure(entry)
        target = None
        for key in ["label", "fitness", "score", "dms_score", "DMS_score"]:
            if key in entry:
                target = float(entry[key])
                break
        return sequence, structure_tokens, entry["mutant"], target

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        sequences, structure_tokens, mutants, targets = tuple(zip(*batch))
        inputs = {"inputs": self._collate_prosst_inputs(sequences, structure_tokens)}
        labels = {"mutants": mutants}
        if any(target is not None for target in targets):
            if not all(target is not None for target in targets):
                raise ValueError("All ProSST mutation samples in a batch need targets or none do.")
            labels["labels"] = torch.tensor(targets, dtype=torch.float)
        return inputs, labels
