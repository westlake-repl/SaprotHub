import torch

from ..data_interface import register_dataset
from .base import ProSSTDatasetBase


@register_dataset
class ProSSTRegressionDataset(ProSSTDatasetBase):
    def __init__(
        self,
        min_clip: list = None,
        mix_max_norm: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_clip = min_clip
        self.mix_max_norm = mix_max_norm

    def __getitem__(self, index):
        entry = self._load_entry(index)
        sequence, structure_tokens = self._load_sequence_and_structure(entry)
        label = float(entry["fitness"])

        if self.min_clip is not None:
            given_min, clip_value = self.min_clip
            if label < given_min:
                label = clip_value

        if self.mix_max_norm is not None:
            min_norm, max_norm = self.mix_max_norm
            label = (label - min_norm) / (max_norm - min_norm)

        return sequence, structure_tokens, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        sequences, structure_tokens, labels = tuple(zip(*batch))
        inputs = {
            "inputs": self._collate_prosst_inputs(sequences, structure_tokens)
        }
        labels = {"labels": torch.tensor(labels, dtype=torch.float)}

        return inputs, labels
