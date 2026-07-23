import gc
import hashlib
import os
import tempfile
import zipfile
from collections import deque
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
ESMFOLD_MIN_TRUNK_CHUNK_SIZE = 8
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
            f"{context} has {len(sequence)} residues, but local ESMFold v1 "
            f"accepts at most {int(max_residues)}. "
            "Use the "
            "sequence + structure files input method for longer proteins."
        )
    return sequence


def _esmfold_cache_path(sequence: str, cache_dir: str) -> Path:
    digest = hashlib.sha256(
        f"{ESMFOLD_CACHE_VERSION}|{sequence}".encode("ascii")
    ).hexdigest()
    return Path(cache_dir) / "esmfold" / f"{digest}.pdb"


def _validate_pdb_output(pdb_text: str) -> None:
    if not pdb_text.strip() or not any(
        line.startswith("ATOM") for line in pdb_text.splitlines()
    ):
        raise ESMFoldPredictionError(
            "ESMFold generated an invalid PDB without protein coordinates."
        )


def _write_cached_pdb(pdb_text: str, output_path: Path) -> None:
    _validate_pdb_output(pdb_text)
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


def _build_esmfold_batches(
    sequences: list[str],
    max_batch_size: int = ESMFOLD_MAX_BATCH_SIZE,
) -> list[list[str]]:
    batches = []
    current = []
    current_max_length = 0
    # Run the most memory-intensive batches first so CUDA reaches its peak
    # early instead of growing its cache throughout the task.
    for sequence in sorted(sequences, key=len, reverse=True):
        next_max_length = max(current_max_length, len(sequence))
        next_size = len(current) + 1
        if (
            current
            and (
                next_size > max_batch_size
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


def _cuda_memory_info(torch, device) -> tuple[int, int, str]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return int(free_bytes), int(total_bytes), "CUDA runtime"
    except Exception:
        try:
            with torch.cuda.device(device_index):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes), int(total_bytes), "CUDA runtime"
        except Exception:
            try:
                total_bytes = int(
                    torch.cuda.get_device_properties(device_index).total_memory
                )
                reserved_bytes = int(torch.cuda.memory_reserved(device_index))
                free_bytes = max(0, total_bytes - reserved_bytes)
                return free_bytes, total_bytes, "conservative PyTorch estimate"
            except Exception as estimate_error:
                raise RuntimeError(
                    "CUDA memory information is unavailable."
                ) from estimate_error


def _adaptive_esmfold_batch_size(torch, device, logger: Callable[[str], None]) -> int:
    if device.type != "cuda":
        return ESMFOLD_MAX_BATCH_SIZE
    try:
        free_bytes, total_bytes, source = _cuda_memory_info(torch, device)
    except Exception:
        logger(
            "Could not read free GPU memory; using the safest ESMFold batch size 1."
        )
        return 1

    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    if free_gib < 8:
        max_batch_size = 1
    elif free_gib < 12:
        max_batch_size = 2
    else:
        max_batch_size = ESMFOLD_MAX_BATCH_SIZE
    logger(
        f"ESMFold GPU memory after model loading: {free_gib:.2f}/{total_gib:.2f} "
        f"GiB free ({source}); adaptive maximum batch size={max_batch_size}."
    )
    return max_batch_size


def _esmfold_chunk_size(
    sequence_length: int,
    free_gpu_bytes: Optional[int] = None,
) -> int:
    if sequence_length > 768:
        chunk_size = ESMFOLD_MIN_TRUNK_CHUNK_SIZE
    elif sequence_length > 512:
        chunk_size = 16
    elif sequence_length > 256:
        chunk_size = 32
    else:
        chunk_size = ESMFOLD_TRUNK_CHUNK_SIZE

    if free_gpu_bytes is not None:
        free_gib = free_gpu_bytes / 1024**3
        if free_gib < 5:
            chunk_size = min(chunk_size, ESMFOLD_MIN_TRUNK_CHUNK_SIZE)
        elif free_gib < 8:
            chunk_size = min(chunk_size, 16)
    return chunk_size


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


def _prepare_esmfold_residue_constants(model, torch, device) -> None:
    structure_module = model.trunk.structure_module
    structure_module._init_residue_constants(torch.float32, device)
    # transformers 4.43 builds this index buffer as int32; torch 2.2 needs int64.
    if structure_module.group_idx.dtype != torch.long:
        structure_module.group_idx = structure_module.group_idx.long()


def _load_local_esmfold_model(model_name: str, device):
    from transformers import AutoConfig, EsmForProteinFolding

    config = AutoConfig.from_pretrained(model_name)
    load_kwargs = {
        "config": config,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        config.esmfold_config.fp16_esm = True
        load_kwargs["device_map"] = {"": str(device)}

    model = EsmForProteinFolding.from_pretrained(model_name, **load_kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.trunk.set_chunk_size(ESMFOLD_TRUNK_CHUNK_SIZE)
    return model.eval()


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
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ESMFoldPredictionError(
            "Local ESMFold v1 requires torch and transformers. Run the "
            "ColabProSST installation cell before using sequence-only input."
        ) from exc

    if not torch.cuda.is_available():
        logger(
            "Local ESMFold v1 is running on CPU. It can require more than 16 GB "
            "of system RAM and is very slow; a Colab GPU runtime is strongly "
            "recommended."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = None
    model = None
    try:
        if device.type == "cuda":
            logger(
                "Loading local ESMFold v1 directly on GPU "
                "(ESM FP16, folding trunk FP32)."
            )
        else:
            logger(f"Loading local ESMFold v1 on CPU: {ESMFOLD_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL)
        model = _load_local_esmfold_model(ESMFOLD_MODEL, device)
        _prepare_esmfold_residue_constants(model, torch, device)

        max_batch_size = _adaptive_esmfold_batch_size(torch, device, logger)
        batches = _build_esmfold_batches(
            missing_sequences,
            max_batch_size=max_batch_size,
        )
        pending_batches = deque(batches)
        cached_count = len(normalized_sequences) - len(missing_sequences)
        logger(
            "Local ESMFold v1: "
            f"{len(missing_sequences)} structure(s) to predict in "
            f"{len(batches)} batch(es); {cached_count} cached."
        )
        completed = 0
        batch_attempt = 0
        while pending_batches:
            batch = pending_batches.popleft()
            batch_attempt += 1
            lengths = [len(sequence) for sequence in batch]
            free_gpu_bytes = None
            free_gpu_note = ""
            if device.type == "cuda":
                try:
                    free_gpu_bytes, _total_gpu_bytes, memory_source = (
                        _cuda_memory_info(torch, device)
                    )
                    free_gpu_note = (
                        f", {free_gpu_bytes / 1024**3:.2f} GiB GPU free "
                        f"({memory_source})"
                    )
                except Exception:
                    free_gpu_note = ", GPU free memory unavailable"
            chunk_size = _esmfold_chunk_size(
                max(lengths),
                free_gpu_bytes=free_gpu_bytes,
            )
            model.trunk.set_chunk_size(chunk_size)
            logger(
                f"ESMFold batch attempt {batch_attempt}: {len(batch)} protein(s), "
                f"{min(lengths)}-{max(lengths)} residues, chunk size={chunk_size}; "
                f"{len(pending_batches)} batch(es) queued{free_gpu_note}."
            )
            while True:
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
                outputs = None
                pdbs = None
                oom_error = None
                try:
                    with torch.inference_mode():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                    pdbs = _convert_esmfold_outputs_to_pdb(outputs, lengths)

                    for sequence, pdb_text in zip(batch, pdbs):
                        _write_cached_pdb(pdb_text, output_paths[sequence])
                        completed += 1
                    logger(
                        f"Local ESMFold v1 progress: {completed}/"
                        f"{len(missing_sequences)} structure(s) saved."
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    oom_error = exc
                finally:
                    del encoded, input_ids, attention_mask, outputs, pdbs
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        allocated_gib = (
                            torch.cuda.memory_allocated(device) / 1024**3
                        )
                        reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
                        logger(
                            "GPU memory after batch cleanup: "
                            f"{allocated_gib:.2f} GiB allocated, "
                            f"{reserved_gib:.2f} GiB reserved."
                        )

                if oom_error is None:
                    break
                if len(batch) > 1:
                    midpoint = (len(batch) + 1) // 2
                    first_half = batch[:midpoint]
                    second_half = batch[midpoint:]
                    pending_batches.appendleft(second_half)
                    pending_batches.appendleft(first_half)
                    logger(
                        "ESMFold GPU memory was insufficient. The batch was "
                        f"automatically split into {len(first_half)} and "
                        f"{len(second_half)} protein(s) for retry."
                    )
                    oom_error = None
                    break
                if chunk_size > ESMFOLD_MIN_TRUNK_CHUNK_SIZE:
                    chunk_size = max(
                        ESMFOLD_MIN_TRUNK_CHUNK_SIZE,
                        chunk_size // 2,
                    )
                    model.trunk.set_chunk_size(chunk_size)
                    logger(
                        "ESMFold GPU memory was insufficient for this protein. "
                        f"Retrying with chunk size={chunk_size}."
                    )
                    oom_error = None
                    continue
                raise ESMFoldPredictionError(
                    "Local ESMFold v1 still ran out of GPU memory for one "
                    f"{lengths[0]}-residue protein at the safest chunk size. "
                    "Use sequence + structure files or a higher-memory GPU."
                ) from oom_error
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
