import torch

from ..data_interface import register_dataset
from .pair_base import ProSSTPairDatasetBase


@register_dataset
class ProSSTPairClassificationDataset(ProSSTPairDatasetBase):
    def __getitem__(self, index):
        entry = self._load_entry(index)
        sequence_1, structure_tokens_1 = self._load_pair_entry(entry, 1)
        sequence_2, structure_tokens_2 = self._load_pair_entry(entry, 2)
        return (
            sequence_1,
            structure_tokens_1,
            sequence_2,
            structure_tokens_2,
            int(entry["label"]),
        )

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        inputs = self._collate_pair_inputs(batch)
        labels = torch.tensor([sample[4] for sample in batch], dtype=torch.long)
        return inputs, {"labels": labels}
