import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from saprot.data.pdb2prosst import (
    load_or_quantize_structure,
    serialize_structure_tokens,
    validate_sequence_and_structure,
)


ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_RESIDUES = 400
ESMFOLD_CACHE_VERSION = "esmfold-v1-api"
SUPPORTED_SEQUENCE_CHARACTERS = frozenset("ACDEFGHIKLMNPQRSTVWYX")


class ESMFoldPredictionError(RuntimeError):
    """Raised when automatic sequence-to-structure prediction fails."""


def normalize_protein_sequence(value, context: str = "sequence") -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise ValueError(f"{context} is empty.")
    invalid = sorted(set(sequence) - SUPPORTED_SEQUENCE_CHARACTERS)
    if invalid:
        raise ValueError(
            f"{context} contains unsupported amino-acid characters: {invalid}. "
            "Automatic structure prediction accepts the 20 standard amino "
            "acids and X."
        )
    if len(sequence) > ESMFOLD_MAX_RESIDUES:
        raise ValueError(
            f"{context} has {len(sequence)} residues, but the automatic "
            f"ESMFold service accepts at most {ESMFOLD_MAX_RESIDUES}. Prepare "
            "a structure-token CSV separately for longer proteins."
        )
    return sequence


def _esmfold_cache_path(sequence: str, cache_dir: str) -> Path:
    digest = hashlib.sha256(
        f"{ESMFOLD_CACHE_VERSION}|{sequence}".encode("ascii")
    ).hexdigest()
    return Path(cache_dir) / "esmfold" / f"{digest}.pdb"


def _validate_pdb_response(pdb_text: str) -> None:
    if not pdb_text.strip() or not any(
        line.startswith("ATOM") for line in pdb_text.splitlines()
    ):
        raise ESMFoldPredictionError(
            "ESMFold returned an invalid response without protein coordinates."
        )


def predict_structure_with_esmfold(
    sequence: str,
    cache_dir: str,
    timeout: int = 300,
    max_retries: int = 2,
    request_post: Optional[Callable] = None,
) -> str:
    sequence = normalize_protein_sequence(sequence)
    output_path = _esmfold_cache_path(sequence, cache_dir)
    if output_path.is_file() and output_path.stat().st_size > 0:
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    post = request_post or requests.post
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = post(
                ESMFOLD_API_URL,
                data=sequence,
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )
            if response.status_code != 200:
                detail = str(getattr(response, "text", "")).strip()[:300]
                raise ESMFoldPredictionError(
                    "ESMFold request failed with HTTP "
                    f"{response.status_code}: {detail or 'no error message'}"
                )
            pdb_text = response.text
            _validate_pdb_response(pdb_text)

            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".pdb.tmp",
                prefix=f"{output_path.stem}-",
                dir=output_path.parent,
                delete=False,
            )
            temp_path = Path(handle.name)
            try:
                with handle:
                    handle.write(pdb_text)
                os.replace(temp_path, output_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            return str(output_path)
        except (requests.RequestException, ESMFoldPredictionError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)

    raise ESMFoldPredictionError(
        "Automatic structure prediction failed after "
        f"{max_retries + 1} attempts. The public ESMFold service may be busy; "
        "retry later or use a prepared structure-token CSV."
    ) from last_error


def prepare_sequence_csv_with_structure_tokens(
    input_csv: str,
    output_csv: str,
    cache_dir: str,
    structure_vocab_size: int,
    pair_mode: bool = False,
    structure_predictor: Callable = predict_structure_with_esmfold,
    structure_quantizer: Callable = load_or_quantize_structure,
) -> str:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    lower_columns = {column.lower(): column for column in df.columns}
    if pair_mode:
        sequence_columns = []
        for index in (1, 2):
            column = lower_columns.get(f"sequence_{index}")
            if column is None:
                raise ValueError(
                    "Sequence-only protein-pair input requires sequence_1 and "
                    "sequence_2 columns."
                )
            sequence_columns.append((column, f"structure_tokens_{index}"))
    else:
        sequence_column = lower_columns.get(
            "sequence",
            lower_columns.get("protein"),
        )
        if sequence_column is None:
            raise ValueError(
                "Sequence-only input requires a sequence column."
            )
        sequence_columns = [(sequence_column, "structure_tokens")]

    normalized = {}
    ordered_sequences = []
    for sequence_column, _token_column in sequence_columns:
        values = []
        for row_index, value in df[sequence_column].items():
            sequence = normalize_protein_sequence(
                value,
                context=f"row {row_index} {sequence_column}",
            )
            values.append(sequence)
            if sequence not in normalized:
                normalized[sequence] = None
                ordered_sequences.append(sequence)
        df[sequence_column] = values

    total = len(ordered_sequences)
    for index, sequence in enumerate(ordered_sequences, start=1):
        print(
            f"Preparing structure {index}/{total}: "
            f"{len(sequence)} residues"
        )
        pdb_path = structure_predictor(sequence, cache_dir=str(cache_dir))
        result = structure_quantizer(
            pdb_path,
            cache_dir=str(cache_dir),
            structure_vocab_size=int(structure_vocab_size),
        )
        tokens = [int(token) for token in result["structure_tokens"]]
        parsed_sequence = str(result.get("sequence") or "").strip().upper()
        validate_sequence_and_structure(
            sequence,
            tokens,
            context=f"automatic structure {index}/{total}",
        )
        if parsed_sequence and parsed_sequence != sequence:
            raise ValueError(
                "The sequence parsed from the predicted structure does not "
                "match the input sequence."
            )
        normalized[sequence] = serialize_structure_tokens(tokens)

    for sequence_column, token_column in sequence_columns:
        df[token_column] = [normalized[sequence] for sequence in df[sequence_column]]
    df["structure_vocab_size"] = int(structure_vocab_size)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("saved prepared structure-token CSV:", output_path)
    return str(output_path)
