from pathlib import Path
from typing import Callable

import pandas as pd

from saprot.data.pdb2prosst import (
    load_or_quantize_structure,
    serialize_structure_tokens,
    validate_sequence_and_structure,
)
from saprot.data.sequence_to_prosst import normalize_protein_sequence


SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}


def _resolve_structure_file(value, structure_dir: str, context: str) -> Path:
    filename = str(value).strip().replace("\\", "/")
    if not filename:
        raise ValueError(f"{context} is empty.")

    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"{context} must be a filename or a relative path inside the "
            "uploaded structure ZIP."
        )

    root = Path(structure_dir).resolve()
    direct_path = (root / relative_path).resolve()
    if root not in [direct_path, *direct_path.parents]:
        raise ValueError(f"{context} points outside the uploaded structure ZIP.")
    if direct_path.is_file():
        resolved = direct_path
    elif len(relative_path.parts) == 1:
        matches = [path for path in root.rglob(filename) if path.is_file()]
        if not matches:
            raise FileNotFoundError(
                f"{context}={filename!r} was not found in the uploaded "
                "structure ZIP."
            )
        if len(matches) > 1:
            choices = [str(path.relative_to(root)) for path in matches[:5]]
            raise ValueError(
                f"{context}={filename!r} is ambiguous because the ZIP contains "
                f"multiple files with that name: {choices}. Use a relative path "
                "such as folder/file.pdb in the CSV."
            )
        resolved = matches[0]
    else:
        raise FileNotFoundError(
            f"{context}={filename!r} was not found in the uploaded structure ZIP."
        )

    if resolved.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
        raise ValueError(
            f"{context} must reference a PDB or mmCIF file, found "
            f"{resolved.name!r}."
        )
    return resolved


def prepare_structure_csv_with_structure_tokens(
    input_csv: str,
    structure_dir: str,
    output_csv: str,
    cache_dir: str,
    structure_vocab_size: int,
    pair_mode: bool = False,
    structure_quantizer: Callable = load_or_quantize_structure,
) -> str:
    """Quantize structures referenced by a task CSV and preserve all columns."""
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    indexes = (1, 2) if pair_mode else (None,)
    inputs = []
    for index in indexes:
        suffix = f"_{index}" if index is not None else ""
        sequence_column = lower_columns.get(f"sequence{suffix}")
        structure_column = lower_columns.get(f"structure_file{suffix}")
        chain_column = lower_columns.get(f"chain{suffix}")
        if sequence_column is None:
            raise ValueError(
                f"Structure-file input requires a sequence{suffix} column."
            )
        if structure_column is None:
            raise ValueError(
                f"Structure-file input requires a structure_file{suffix} column."
            )
        inputs.append(
            (sequence_column, structure_column, chain_column, f"structure_tokens{suffix}")
        )

    result_cache = {}
    total = len(df) * len(inputs)
    completed = 0
    for sequence_column, structure_column, chain_column, token_column in inputs:
        normalized_sequences = []
        serialized_tokens = []
        for row_index, row in df.iterrows():
            completed += 1
            context = f"row {row_index + 2} {structure_column}"
            sequence = normalize_protein_sequence(
                row[sequence_column],
                context=f"row {row_index + 2} {sequence_column}",
                max_residues=None,
            )
            structure_path = _resolve_structure_file(
                row[structure_column],
                structure_dir,
                context,
            )
            chain_id = None
            if chain_column is not None and not pd.isna(row[chain_column]):
                chain_id = str(row[chain_column]).strip() or None
            cache_key = (str(structure_path), chain_id)
            if cache_key not in result_cache:
                print(
                    f"Preparing structure {completed}/{total}: "
                    f"{structure_path.name}"
                )
                result_cache[cache_key] = structure_quantizer(
                    str(structure_path),
                    cache_dir=str(cache_dir),
                    chain_id=chain_id,
                    structure_vocab_size=int(structure_vocab_size),
                )
            result = result_cache[cache_key]
            tokens = [int(token) for token in result["structure_tokens"]]
            parsed_sequence = "".join(
                str(result.get("sequence") or "").split()
            ).upper()
            validate_sequence_and_structure(sequence, tokens, context=context)
            if (
                parsed_sequence
                and len(parsed_sequence) == len(sequence)
                and all(
                    csv_residue == structure_residue or csv_residue == "X"
                    for csv_residue, structure_residue in zip(
                        sequence,
                        parsed_sequence,
                    )
                )
            ):
                restored_positions = [
                    index + 1
                    for index, (csv_residue, structure_residue) in enumerate(
                        zip(sequence, parsed_sequence)
                    )
                    if csv_residue == "X" and structure_residue != "X"
                ]
                if restored_positions:
                    print(
                        f"Restored {len(restored_positions)} X residue(s) in "
                        f"row {row_index + 2} from the supplied structure: "
                        f"positions {restored_positions}."
                    )
                    completed_sequence = list(sequence)
                    for position in restored_positions:
                        completed_sequence[position - 1] = parsed_sequence[
                            position - 1
                        ]
                    sequence = "".join(completed_sequence)
            if parsed_sequence and parsed_sequence != sequence:
                raise ValueError(
                    f"{context} does not match {sequence_column}: the CSV has "
                    f"{len(sequence)} residues, but the selected structure "
                    "contains a different amino-acid sequence. Check the file "
                    "name and chain column."
                )
            normalized_sequences.append(sequence)
            serialized_tokens.append(serialize_structure_tokens(tokens))

        df[sequence_column] = normalized_sequences
        df[token_column] = serialized_tokens

    df["structure_vocab_size"] = int(structure_vocab_size)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Structure tokens are ready for this task.")
    return str(output_path)
