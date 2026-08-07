import torch

from ..data_interface import register_dataset
from .base import ProSSTDatasetBase

try:
    from saprot.data.prosst_labels import (
        RESIDUE_LABEL_IGNORE_INDEX,
        parse_residue_labels,
        validate_residue_labels,
    )
except ImportError:
    from data.prosst_labels import (
        RESIDUE_LABEL_IGNORE_INDEX,
        parse_residue_labels,
        validate_residue_labels,
    )


@register_dataset
class ProSSTTokenClassificationDataset(ProSSTDatasetBase):
    def __getitem__(self, index):
        entry = self._load_entry(index)
        full_sequence = entry["seq"].strip().upper()
        labels = parse_residue_labels(entry["label"])
        validate_residue_labels(full_sequence, labels, context=f"sample {index}")

        sequence, structure_tokens = self._load_sequence_and_structure(entry)
        labels = labels[: len(sequence)]
        validate_residue_labels(sequence, labels, context=f"sample {index}")
        return sequence, structure_tokens, labels

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        sequences, structure_tokens, residue_labels = tuple(zip(*batch))
        encoder_info = self._collate_prosst_inputs(sequences, structure_tokens)
        target_length = encoder_info["input_ids"].shape[1]

        labels = torch.full(
            (len(batch), target_length),
            RESIDUE_LABEL_IGNORE_INDEX,
            dtype=torch.long,
        )
        for row_idx, (sequence, sample_labels) in enumerate(
            zip(sequences, residue_labels)
        ):
            encoded_length = int(encoder_info["attention_mask"][row_idx].sum())
            expected_length = len(sequence) + 2
            if encoded_length != expected_length:
                raise ValueError(
                    "ProSST tokenizer must produce one token per residue plus "
                    "CLS/EOS for residue-level classification: "
                    f"encoded={encoded_length}, expected={expected_length}."
                )
            labels[row_idx, 1 : 1 + len(sample_labels)] = torch.tensor(
                sample_labels,
                dtype=torch.long,
            )

        inputs = {"inputs": encoder_info}
        return inputs, {"labels": labels}
