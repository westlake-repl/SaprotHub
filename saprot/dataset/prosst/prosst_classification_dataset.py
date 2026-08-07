import torch

from ..data_interface import register_dataset
from .base import ProSSTDatasetBase


@register_dataset
class ProSSTClassificationDataset(ProSSTDatasetBase):
    def __init__(self, preset_label: int = None, **kwargs):
        super().__init__(**kwargs)
        self.preset_label = preset_label

    def __getitem__(self, index):
        entry = self._load_entry(index)
        sequence, structure_tokens = self._load_sequence_and_structure(entry)
        label = entry["label"] if self.preset_label is None else self.preset_label

        return sequence, structure_tokens, int(label)

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        sequences, structure_tokens, label_ids = tuple(zip(*batch))
        inputs = {
            "inputs": self._collate_prosst_inputs(sequences, structure_tokens)
        }
        labels = {"labels": torch.tensor(label_ids, dtype=torch.long)}

        return inputs, labels
