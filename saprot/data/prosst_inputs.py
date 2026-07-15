from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch

from .pdb2prosst import encode_structure_tokens, pad_structure_input_ids


def find_sequence_column(columns: Sequence[str]) -> str:
    lower_columns = {column.lower(): column for column in columns}
    if "sequence" in lower_columns:
        return lower_columns["sequence"]
    if "protein" in lower_columns:
        return lower_columns["protein"]
    raise ValueError("ProSST CSV must contain `sequence` or `protein`.")


def structure_entry_from_row(
    row: pd.Series,
    csv_dir: Path,
    structure_base_dir: str = None,
    pair_index: Optional[int] = None,
) -> Dict[str, Any]:
    lower_columns = {column.lower(): column for column in row.index}
    suffix = f"_{pair_index}" if pair_index is not None else ""
    structure_column = lower_columns.get(f"structure_tokens{suffix}")
    vocab_columns = [f"structure_vocab_size{suffix}"]
    if pair_index is not None:
        vocab_columns.append("structure_vocab_size")

    def add_vocab_metadata(entry):
        for name in vocab_columns:
            vocab_column = lower_columns.get(name)
            if vocab_column is not None and _has_value(row[vocab_column]):
                entry["structure_vocab_size"] = row[vocab_column]
                break
        return entry

    if structure_column is not None and _has_value(row[structure_column]):
        return add_vocab_metadata(
            {"structure_tokens": row[structure_column]}
        )

    path_column = None
    for name in [f"structure_path{suffix}", f"pdb_path{suffix}"]:
        if name in lower_columns and _has_value(row[lower_columns[name]]):
            path_column = lower_columns[name]
            break
    if path_column is None:
        subject = f"pair protein {pair_index}" if pair_index is not None else "row"
        raise ValueError(
            f"Each ProSST {subject} needs structure_tokens{suffix}, "
            f"structure_path{suffix}, or pdb_path{suffix}."
        )

    structure_path = Path(str(row[path_column]).strip())
    if not structure_path.is_absolute():
        candidates = [csv_dir / structure_path]
        if structure_base_dir is not None and str(structure_base_dir).strip():
            candidates.append(Path(structure_base_dir) / structure_path)
        structure_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[-1],
        )

    entry = {"pdb_path": str(structure_path)}
    for name in [f"chain_id{suffix}", f"chain{suffix}"]:
        column = lower_columns.get(name)
        if column is not None and _has_value(row[column]):
            entry["chain_id"] = str(row[column]).strip()
            break
    return add_vocab_metadata(entry)


def prepare_prosst_batch(
    tokenizer,
    sequences: Sequence[str],
    structure_tokens_list: Sequence[Sequence[int]],
    max_length: int,
    structure_vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    truncated_sequences = [sequence[:max_length] for sequence in sequences]
    truncated_structures = [
        tokens[:max_length] for tokens in structure_tokens_list
    ]
    inputs = tokenizer.batch_encode_plus(
        truncated_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length + 2,
    )
    target_length = inputs["input_ids"].shape[1]
    structure_input_ids = [
        encode_structure_tokens(tokens, structure_vocab_size)
        for tokens in truncated_structures
    ]
    inputs["ss_input_ids"] = torch.tensor(
        pad_structure_input_ids(structure_input_ids, target_length),
        dtype=torch.long,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _has_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""
