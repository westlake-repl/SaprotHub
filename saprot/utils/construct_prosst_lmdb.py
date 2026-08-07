import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import lmdb
import pandas as pd
from tqdm import tqdm

_SAPROT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SAPROT_DIR.parent
for _path in [str(_REPO_ROOT), str(_SAPROT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from saprot.data.pdb2prosst import (
        get_structure_tokens_from_entry,
        serialize_structure_tokens,
        validate_sequence_and_structure,
    )
    from saprot.data.prosst_labels import (
        parse_residue_labels,
        validate_residue_labels,
    )
except ImportError:
    from data.pdb2prosst import (
        get_structure_tokens_from_entry,
        serialize_structure_tokens,
        validate_sequence_and_structure,
    )
    from data.prosst_labels import parse_residue_labels, validate_residue_labels


VALID_STAGES = {"train", "valid", "test"}
PAIR_TASK_TYPES = {"pair_classification", "pair_regression"}
MIN_LMDB_MAP_SIZE = 1 << 26


def _has_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _estimate_lmdb_map_size(data_dict: Dict[Any, Any]) -> int:
    payload_size = 0
    for key, value in data_dict.items():
        payload_size += len(str(key).encode()) + len(str(value).encode())

    return max(MIN_LMDB_MAP_SIZE, payload_size * 16)


def _parse_classification_label(value: Any, context: str) -> int:
    try:
        numeric_value = float(value)
        label = int(numeric_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} label must be an integer category ID.") from exc
    if not math.isfinite(numeric_value) or numeric_value != label or label < 0:
        raise ValueError(f"{context} label must be a non-negative integer category ID.")
    return label


def _dump_lmdb(data_dict: Dict[Any, Any], lmdb_dir: str) -> None:
    os.makedirs(lmdb_dir, exist_ok=True)
    env = lmdb.open(lmdb_dir, map_size=_estimate_lmdb_map_size(data_dict))

    try:
        with env.begin(write=True) as operator:
            for key, value in tqdm(data_dict.items(), desc="Dumping data..."):
                operator.put(key=str(key).encode(), value=str(value).encode())
    finally:
        env.close()


def _resolve_path(
    path_value: Any,
    csv_dir: Path,
    structure_base_dir: Optional[str] = None,
) -> str:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return str(path)

    candidates = [csv_dir / path]
    if structure_base_dir is not None and str(structure_base_dir).strip():
        candidates.append(Path(structure_base_dir) / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[-1])


def _get_sequence_column(df: pd.DataFrame) -> str:
    if "sequence" in df.columns:
        return "sequence"
    if "protein" in df.columns:
        return "protein"
    raise ValueError(
        "ProSST CSV must contain a sequence column named `sequence` "
        "(or `protein` for SaprotHub-style CSVs)."
    )


def _get_label_column(df: pd.DataFrame, task_type: str) -> str:
    if task_type == "token_classification":
        if "residue_labels" in df.columns:
            return "residue_labels"
        if "label" in df.columns:
            return "label"
        raise ValueError(
            "ProSST residue-level classification CSV must contain "
            "`residue_labels` (or `label`)."
        )
    if task_type == "regression" and "fitness" in df.columns:
        return "fitness"
    if "label" in df.columns:
        return "label"
    if task_type == "regression":
        raise ValueError("ProSST regression CSV must contain `label` or `fitness`.")
    raise ValueError("ProSST classification CSV must contain `label`.")


def _get_structure_path_column(row: pd.Series) -> Optional[str]:
    for column in ["structure_path", "pdb_path"]:
        if column in row.index and _has_value(row[column]):
            return column
    return None


def _get_pair_structure_entry(
    row: pd.Series,
    pair_index: int,
    csv_dir: Path,
    structure_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    entry = {}
    vocab_column = f"structure_vocab_size_{pair_index}"
    if vocab_column in row.index and _has_value(row[vocab_column]):
        entry["structure_vocab_size"] = row[vocab_column]
    elif "structure_vocab_size" in row.index and _has_value(
        row["structure_vocab_size"]
    ):
        entry["structure_vocab_size"] = row["structure_vocab_size"]

    token_column = f"structure_tokens_{pair_index}"
    if token_column in row.index and _has_value(row[token_column]):
        entry["structure_tokens"] = row[token_column]
        return entry

    path_column = next(
        (
            column
            for column in [
                f"structure_path_{pair_index}",
                f"pdb_path_{pair_index}",
            ]
            if column in row.index and _has_value(row[column])
        ),
        None,
    )
    if path_column is None:
        raise ValueError(
            f"Pair protein {pair_index} must contain structure_tokens_"
            f"{pair_index}, structure_path_{pair_index}, or "
            f"pdb_path_{pair_index}."
        )

    entry["pdb_path"] = _resolve_path(
        row[path_column],
        csv_dir,
        structure_base_dir=structure_base_dir,
    )
    for chain_column in [f"chain_id_{pair_index}", f"chain_{pair_index}"]:
        if chain_column in row.index and _has_value(row[chain_column]):
            entry["chain_id"] = str(row[chain_column]).strip()
            break
    return entry


def _build_sample(
    row: pd.Series,
    csv_dir: Path,
    sequence_column: str,
    label_column: str,
    task_type: str,
    cache_dir: Optional[str],
    structure_vocab_size: int,
    structure_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    sequence = str(row[sequence_column]).strip().upper()
    if not sequence:
        raise ValueError("ProSST CSV contains an empty sequence.")

    entry = {}
    if "structure_vocab_size" in row.index and _has_value(
        row["structure_vocab_size"]
    ):
        entry["structure_vocab_size"] = row["structure_vocab_size"]
    if "structure_tokens" in row.index and _has_value(row["structure_tokens"]):
        entry["structure_tokens"] = row["structure_tokens"]
    else:
        structure_path_column = _get_structure_path_column(row)
        if structure_path_column is None:
            raise ValueError(
                "ProSST CSV must contain `structure_tokens`, `structure_path`, "
                "or `pdb_path`."
            )

        entry["pdb_path"] = _resolve_path(
            row[structure_path_column],
            csv_dir,
            structure_base_dir=structure_base_dir,
        )
        if "chain_id" in row.index and _has_value(row["chain_id"]):
            entry["chain_id"] = str(row["chain_id"]).strip()
        elif "chain" in row.index and _has_value(row["chain"]):
            entry["chain_id"] = str(row["chain"]).strip()

    structure_tokens = get_structure_tokens_from_entry(
        entry,
        cache_dir=cache_dir,
        structure_vocab_size=structure_vocab_size,
    )
    validate_sequence_and_structure(sequence, structure_tokens, context=sequence[:20])

    label = row[label_column]
    if task_type == "classification":
        label = _parse_classification_label(label, "Classification")
        label_key = "label"
    elif task_type == "regression":
        label = float(label)
        label_key = "fitness"
    elif task_type == "token_classification":
        label = parse_residue_labels(label)
        validate_residue_labels(sequence, label, context=sequence[:20])
        label_key = "label"
    else:
        raise ValueError(
            "ProSST LMDB construction supports classification, regression, "
            "and token_classification."
        )

    sample = {
        "seq": sequence,
        label_key: label,
        "structure_tokens": serialize_structure_tokens(structure_tokens),
        "structure_vocab_size": int(structure_vocab_size),
    }
    if "pdb_path" in entry:
        sample["pdb_path"] = entry["pdb_path"]
    if "chain_id" in entry:
        sample["chain_id"] = entry["chain_id"]

    return sample


def _build_pair_sample(
    row: pd.Series,
    csv_dir: Path,
    task_type: str,
    cache_dir: Optional[str],
    structure_vocab_size: int,
    structure_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    sequences = []
    structure_entries = []
    structure_tokens_list = []
    for pair_index in [1, 2]:
        sequence_column = f"sequence_{pair_index}"
        sequence = str(row[sequence_column]).strip().upper()
        if not sequence or sequence == "NAN":
            raise ValueError(f"Pair protein {pair_index} has an empty sequence.")

        entry = _get_pair_structure_entry(
            row,
            pair_index,
            csv_dir,
            structure_base_dir=structure_base_dir,
        )
        structure_tokens = get_structure_tokens_from_entry(
            entry,
            cache_dir=cache_dir,
            structure_vocab_size=structure_vocab_size,
        )
        validate_sequence_and_structure(
            sequence,
            structure_tokens,
            context=f"pair protein {pair_index}",
        )
        sequences.append(sequence)
        structure_entries.append(entry)
        structure_tokens_list.append(structure_tokens)

    label = row["label"]
    if task_type == "pair_classification":
        label = _parse_classification_label(label, "Pair classification")
    elif task_type == "pair_regression":
        label = float(label)
    else:
        raise ValueError(f"Unsupported pair task_type: {task_type}.")

    sample = {
        "seq_1": sequences[0],
        "seq_2": sequences[1],
        "label": label,
        "structure_tokens_1": serialize_structure_tokens(
            structure_tokens_list[0]
        ),
        "structure_tokens_2": serialize_structure_tokens(
            structure_tokens_list[1]
        ),
        "structure_vocab_size": int(structure_vocab_size),
    }
    for pair_index, entry in enumerate(structure_entries, start=1):
        if "pdb_path" in entry:
            sample[f"pdb_path_{pair_index}"] = entry["pdb_path"]
        if "chain_id" in entry:
            sample[f"chain_id_{pair_index}"] = entry["chain_id"]
        name_column = f"name_{pair_index}"
        if name_column in row.index and _has_value(row[name_column]):
            sample[name_column] = str(row[name_column]).strip()
    return sample


def construct_prosst_lmdb(
    csv_file: str,
    root_dir: str,
    dataset_name: str,
    task_type: str,
    cache_dir: Optional[str] = None,
    structure_vocab_size: int = 2048,
    structure_base_dir: Optional[str] = None,
) -> None:
    """
    Construct a ProSST LMDB dataset from a CSV file.

    Required columns:
        sequence,stage plus structure_tokens, structure_path, or pdb_path.
        classification uses label; regression uses label or fitness;
        token_classification uses residue_labels or label. Pair tasks use
        sequence_1, sequence_2, label, and indexed structure columns.
    """
    if task_type not in {
        "classification",
        "regression",
        "token_classification",
        "pair_classification",
        "pair_regression",
    }:
        raise ValueError(
            "Unsupported ProSST LMDB task_type. Expected classification, "
            "regression, token_classification, pair_classification, or "
            "pair_regression."
        )

    csv_path = Path(csv_file)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    if task_type in PAIR_TASK_TYPES:
        missing_sequence_columns = {
            column for column in ["sequence_1", "sequence_2"] if column not in df
        }
        if missing_sequence_columns:
            raise ValueError(
                "ProSST pair CSV missing required columns: "
                f"{sorted(missing_sequence_columns)}."
            )
        if "label" not in df.columns:
            raise ValueError("ProSST pair CSV must contain `label`.")
        sequence_column = None
        label_column = None
    else:
        sequence_column = _get_sequence_column(df)
        label_column = _get_label_column(df, task_type)
    if "stage" not in df.columns:
        raise ValueError("ProSST CSV missing required column: stage")
    if task_type not in PAIR_TASK_TYPES:
        if (
            "structure_tokens" not in df.columns
            and "structure_path" not in df.columns
            and "pdb_path" not in df.columns
        ):
            raise ValueError(
                "ProSST CSV requires `structure_tokens`, `structure_path`, "
                "or `pdb_path`."
            )

    stages = set(str(stage).strip().lower() for stage in df["stage"].tolist())
    invalid = stages - VALID_STAGES
    if invalid:
        raise ValueError(
            f"Invalid stage values in ProSST CSV: {sorted(invalid)}. "
            f"Expected only {sorted(VALID_STAGES)}."
        )

    data_dicts = {"train": {}, "valid": {}, "test": {}}
    csv_dir = csv_path.parent

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Constructing ProSST LMDB"):
        stage = str(row["stage"]).strip().lower()
        if task_type in PAIR_TASK_TYPES:
            sample = _build_pair_sample(
                row,
                csv_dir=csv_dir,
                task_type=task_type,
                cache_dir=cache_dir,
                structure_vocab_size=structure_vocab_size,
                structure_base_dir=structure_base_dir,
            )
        else:
            sample = _build_sample(
                row,
                csv_dir=csv_dir,
                sequence_column=sequence_column,
                label_column=label_column,
                task_type=task_type,
                cache_dir=cache_dir,
                structure_vocab_size=structure_vocab_size,
                structure_base_dir=structure_base_dir,
            )
        data_dicts[stage][len(data_dicts[stage])] = json.dumps(sample)

    for stage, tmp_dict in data_dicts.items():
        tmp_dict["length"] = len(tmp_dict)
        lmdb_dir = Path(root_dir) / dataset_name / stage
        _dump_lmdb(tmp_dict, str(lmdb_dir))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument(
        "--task_type",
        required=True,
        choices=[
            "classification",
            "regression",
            "token_classification",
            "pair_classification",
            "pair_regression",
        ],
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=2048)
    parser.add_argument("--structure_base_dir", default=None)
    return parser.parse_args()


def main():
    args = get_args()
    construct_prosst_lmdb(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        task_type=args.task_type,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        structure_base_dir=args.structure_base_dir,
    )


if __name__ == "__main__":
    main()
