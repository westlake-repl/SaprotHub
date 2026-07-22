import gc
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from saprot.data.pdb2prosst import (
    load_or_quantize_structure,
    serialize_structure_tokens,
    validate_sequence_and_structure,
)
from saprot.data.sequence_completion import complete_unknown_residues


ESMFOLD_MODEL = "facebook/esmfold_v1"
ESMFOLD_MAX_RESIDUES = 1024
ESMFOLD_CACHE_VERSION = "esmfold-v1-local"
ESMFOLD_TRUNK_CHUNK_SIZE = 64
ESMFOLD_BATCH_SQUARE_BUDGET = ESMFOLD_MAX_RESIDUES**2
ESMFOLD_MAX_BATCH_SIZE = 4
SUPPORTED_SEQUENCE_CHARACTERS = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def preparation_artifact_paths(output_csv: str) -> dict[str, Path]:
    output_path = Path(output_csv)
    stem = output_path.stem
    return {
        "generated_structure_zip": output_path.with_name(
            f"{stem}_generated_structures.zip"
        ),
        "reusable_structure_input_csv": output_path.with_name(
            f"{stem}_reusable_structure_input.csv"
        ),
        "completed_sequences_csv": output_path.with_name(
            f"{stem}_completed_sequences.csv"
        ),
        "x_completion_report_csv": output_path.with_name(
            f"{stem}_x_completion_report.csv"
        ),
    }


def clear_preparation_artifacts(output_csv: str) -> dict[str, Path]:
    paths = preparation_artifact_paths(output_csv)
    for path in paths.values():
        if path.is_file():
            path.unlink()
    return paths


class ESMFoldPredictionError(RuntimeError):
    """Raised when automatic sequence-to-structure prediction fails."""


def normalize_protein_sequence(
    value,
    context: str = "sequence",
    max_residues: Optional[int] = ESMFOLD_MAX_RESIDUES,
) -> str:
    if value is None or bool(pd.isna(value)):
        raise ValueError(f"{context} is empty.")
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
    if max_residues is not None and len(sequence) > int(max_residues):
        raise ValueError(
            f"{context} has {len(sequence)} residues, but the automatic "
            f"local ESMFold v1 workflow accepts at most {int(max_residues)}. "
            "Use the "
            "sequence + structure files input method for longer proteins."
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


def _write_cached_pdb(pdb_text: str, output_path: Path) -> None:
    _validate_pdb_response(pdb_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def _build_esmfold_batches(sequences: list[str]) -> list[list[str]]:
    batches = []
    current = []
    current_max_length = 0
    for sequence in sorted(sequences, key=len):
        next_max_length = max(current_max_length, len(sequence))
        next_size = len(current) + 1
        if (
            current
            and (
                next_size > ESMFOLD_MAX_BATCH_SIZE
                or next_size * next_max_length * next_max_length
                > ESMFOLD_BATCH_SQUARE_BUDGET
            )
        ):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(sequence)
        current_max_length = max(current_max_length, len(sequence))
    if current:
        batches.append(current)
    return batches


def _convert_esmfold_outputs_to_pdb(outputs, sequence_lengths: list[int]) -> list[str]:
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import (
        Protein as OFProtein,
        to_pdb,
    )

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    final_atom_positions = final_atom_positions.detach().cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"].detach().cpu().numpy()
    aatype = outputs["aatype"].detach().cpu().numpy()
    residue_index = outputs["residue_index"].detach().cpu().numpy()
    plddt = outputs["plddt"].detach().float().cpu().numpy() * 100
    chain_index = outputs.get("chain_index")
    if chain_index is not None:
        chain_index = chain_index.detach().cpu().numpy()

    pdbs = []
    for index, length in enumerate(sequence_lengths):
        prediction = OFProtein(
            aatype=aatype[index, :length],
            atom_positions=final_atom_positions[index, :length],
            atom_mask=final_atom_mask[index, :length],
            residue_index=residue_index[index, :length] + 1,
            b_factors=plddt[index, :length],
            chain_index=(
                chain_index[index, :length] if chain_index is not None else None
            ),
        )
        pdbs.append(to_pdb(prediction))
    return pdbs


def predict_structures_with_esmfold(
    sequences: list[str],
    cache_dir: str,
    logger: Callable[[str], None] = print,
) -> dict[str, str]:
    normalized_sequences = list(
        dict.fromkeys(normalize_protein_sequence(sequence) for sequence in sequences)
    )
    output_paths = {
        sequence: _esmfold_cache_path(sequence, cache_dir)
        for sequence in normalized_sequences
    }
    missing_sequences = [
        sequence
        for sequence in normalized_sequences
        if not output_paths[sequence].is_file()
        or output_paths[sequence].stat().st_size == 0
    ]
    if not missing_sequences:
        logger(
            f"Local ESMFold v1: reused {len(normalized_sequences)} cached "
            "structure(s)."
        )
        return {sequence: str(path) for sequence, path in output_paths.items()}

    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as exc:
        raise ESMFoldPredictionError(
            "Local ESMFold v1 requires torch and transformers. Run the "
            "ColabProSST installation cell before using sequence-only input."
        ) from exc

    if not torch.cuda.is_available():
        logger(
            "Local ESMFold v1 is running on CPU. This is supported but can be "
            "very slow; a Colab GPU runtime is strongly recommended."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = None
    model = None
    try:
        logger(
            f"Loading local ESMFold v1 on {device.type}: {ESMFOLD_MODEL}"
        )
        tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL)
        model = EsmForProteinFolding.from_pretrained(
            ESMFOLD_MODEL,
            low_cpu_mem_usage=True,
        )
        if device.type == "cuda":
            model.esm = model.esm.half()
        model.trunk.set_chunk_size(ESMFOLD_TRUNK_CHUNK_SIZE)
        model = model.to(device).eval()

        batches = _build_esmfold_batches(missing_sequences)
        cached_count = len(normalized_sequences) - len(missing_sequences)
        logger(
            "Local ESMFold v1: "
            f"{len(missing_sequences)} structure(s) to predict in "
            f"{len(batches)} batch(es); {cached_count} cached."
        )
        completed = 0
        for batch_index, batch in enumerate(batches, start=1):
            lengths = [len(sequence) for sequence in batch]
            logger(
                f"ESMFold batch {batch_index}/{len(batches)}: "
                f"{len(batch)} protein(s), {min(lengths)}-{max(lengths)} residues."
            )
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            try:
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                pdbs = _convert_esmfold_outputs_to_pdb(outputs, lengths)
            except torch.cuda.OutOfMemoryError as exc:
                raise ESMFoldPredictionError(
                    "Local ESMFold v1 ran out of GPU memory. Retry with a "
                    "higher-memory Colab GPU, or use sequence + structure "
                    "files for this protein."
                ) from exc
            finally:
                del input_ids, attention_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for sequence, pdb_text in zip(batch, pdbs):
                _write_cached_pdb(pdb_text, output_paths[sequence])
                completed += 1
            logger(
                f"Local ESMFold v1 progress: {completed}/"
                f"{len(missing_sequences)} structure(s) saved."
            )
            del outputs, pdbs
    except ESMFoldPredictionError:
        raise
    except Exception as exc:
        raise ESMFoldPredictionError(
            "Local ESMFold v1 structure prediction failed. Retry in a fresh "
            "GPU runtime, or use sequence + structure files."
        ) from exc
    finally:
        del model, tokenizer
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger("Local ESMFold v1 finished and released its model memory.")
    return {sequence: str(path) for sequence, path in output_paths.items()}


def predict_structure_with_esmfold(
    sequence: str,
    cache_dir: str,
) -> str:
    sequence = normalize_protein_sequence(sequence)
    return predict_structures_with_esmfold(
        [sequence],
        cache_dir=cache_dir,
    )[sequence]


def prepare_sequence_csv_with_structure_tokens(
    input_csv: str,
    output_csv: str,
    cache_dir: str,
    structure_vocab_size: int,
    pair_mode: bool = False,
    structure_predictor: Callable = predict_structure_with_esmfold,
    structure_quantizer: Callable = load_or_quantize_structure,
    sequence_completer: Callable = complete_unknown_residues,
) -> str:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    original_columns = list(df.columns)
    artifact_paths = clear_preparation_artifacts(output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

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

    completion_map, completion_audit = sequence_completer(
        ordered_sequences,
        cache_dir=str(cache_dir),
    )
    report_path = artifact_paths["x_completion_report_csv"]
    if completion_audit:
        for sequence_column, _token_column in sequence_columns:
            df[sequence_column] = [
                completion_map[sequence] for sequence in df[sequence_column]
            ]
        ordered_sequences = list(
            dict.fromkeys(completion_map[sequence] for sequence in ordered_sequences)
        )
        normalized = {sequence: None for sequence in ordered_sequences}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(completion_audit).to_csv(report_path, index=False)
        print("X-completion audit report:", report_path)
    total = len(ordered_sequences)
    if structure_predictor is predict_structure_with_esmfold:
        predicted_structure_paths = {
            sequence: Path(path)
            for sequence, path in predict_structures_with_esmfold(
                ordered_sequences,
                cache_dir=str(cache_dir),
            ).items()
        }
    else:
        predicted_structure_paths = {}
        for index, sequence in enumerate(ordered_sequences, start=1):
            print(
                f"Preparing structure {index}/{total}: "
                f"{len(sequence)} residues"
            )
            predicted_structure_paths[sequence] = Path(
                structure_predictor(sequence, cache_dir=str(cache_dir))
            )

    for index, sequence in enumerate(ordered_sequences, start=1):
        print(f"Generating ProSST tokens {index}/{total}.")
        pdb_path = predicted_structure_paths[sequence]
        if not pdb_path.is_file():
            raise FileNotFoundError(
                f"ESMFold structure file does not exist: {pdb_path}"
            )
        predicted_structure_paths[sequence] = pdb_path
        result = structure_quantizer(
            str(pdb_path),
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
    df.to_csv(output_path, index=False)

    user_input_df = df[original_columns].copy()
    if completion_audit:
        user_input_df.to_csv(
            artifact_paths["completed_sequences_csv"],
            index=False,
        )
        print(
            "Completed sequence CSV:",
            artifact_paths["completed_sequences_csv"],
        )

    structure_names = {
        sequence: f"prosst_structure_{index:04d}.pdb"
        for index, sequence in enumerate(ordered_sequences, start=1)
    }
    reusable_df = user_input_df.copy()
    for sequence_column, _token_column in sequence_columns:
        suffix = sequence_column[len("sequence"):]
        reusable_df[f"structure_file{suffix}"] = [
            structure_names[sequence] for sequence in reusable_df[sequence_column]
        ]
    reusable_df.to_csv(
        artifact_paths["reusable_structure_input_csv"],
        index=False,
    )
    with zipfile.ZipFile(
        artifact_paths["generated_structure_zip"],
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for sequence in ordered_sequences:
            archive.write(
                predicted_structure_paths[sequence],
                arcname=structure_names[sequence],
            )
    print(
        "Reusable structure-input CSV:",
        artifact_paths["reusable_structure_input_csv"],
    )
    print(
        "Generated structure ZIP:",
        artifact_paths["generated_structure_zip"],
    )
    print("Structure tokens are ready for this task.")
    return str(output_path)
