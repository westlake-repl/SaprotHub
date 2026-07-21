import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional


ESM2_COMPLETION_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM2_COMPLETION_CACHE_VERSION = "esm2-650m-x-completion-v1"
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_X_COUNT_THRESHOLD = 10
HIGH_X_RATIO_THRESHOLD = 0.05
MAX_LOGGED_PREDICTIONS = 20


def _cache_path(sequence: str, cache_dir: str, model_name: str) -> Path:
    digest = hashlib.sha256(
        (
            f"{ESM2_COMPLETION_CACHE_VERSION}|{model_name}|{sequence}"
        ).encode("ascii")
    ).hexdigest()
    return Path(cache_dir) / "sequence_completion" / f"{digest}.json"


def _load_cached_completion(
    sequence: str,
    cache_dir: str,
    model_name: str,
) -> Optional[dict]:
    path = _cache_path(sequence, cache_dir, model_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("cache_version") != ESM2_COMPLETION_CACHE_VERSION
        or payload.get("model") != model_name
        or payload.get("original_sequence") != sequence
    ):
        return None
    completed = str(payload.get("completed_sequence", ""))
    predictions = payload.get("predictions")
    if (
        len(completed) != len(sequence)
        or "X" in completed
        or not isinstance(predictions, list)
    ):
        return None
    return payload


def _write_cached_completion(
    payload: dict,
    cache_dir: str,
    model_name: str,
) -> None:
    path = _cache_path(payload["original_sequence"], cache_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json.tmp",
        prefix=f"{path.stem}-",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _build_batches(
    sequences: Iterable[str],
    max_batch_size: int,
    max_padded_residues: int,
) -> list[list[str]]:
    ordered = sorted(sequences, key=lambda item: (len(item), item))
    batches = []
    current = []
    current_max_length = 0
    for sequence in ordered:
        next_max_length = max(current_max_length, len(sequence))
        next_size = len(current) + 1
        exceeds_size = next_size > max_batch_size
        exceeds_residues = next_max_length * next_size > max_padded_residues
        if current and (exceeds_size or exceeds_residues):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(sequence)
        current_max_length = max(current_max_length, len(sequence))
    if current:
        batches.append(current)
    return batches


def _predict_batch(
    model,
    tokenizer,
    sequences: list[str],
    device: str,
    model_name: str,
) -> list[dict]:
    import torch

    masked_sequences = [
        sequence.replace("X", tokenizer.mask_token) for sequence in sequences
    ]
    encoded = tokenizer(
        masked_sequences,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    with torch.inference_mode():
        logits = model(**model_inputs).logits

    amino_acid_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(STANDARD_AMINO_ACIDS)),
        device=logits.device,
        dtype=torch.long,
    )
    results = []
    for row_index, sequence in enumerate(sequences):
        mask_positions = torch.nonzero(
            model_inputs["input_ids"][row_index] == tokenizer.mask_token_id,
            as_tuple=False,
        ).flatten()
        sequence_positions = [
            index for index, residue in enumerate(sequence) if residue == "X"
        ]
        if len(mask_positions) != len(sequence_positions):
            raise RuntimeError(
                "ESM-2 tokenization did not preserve every X position in the "
                "protein sequence."
            )

        mask_logits = logits[row_index, mask_positions].float()
        probabilities = torch.softmax(mask_logits, dim=-1)
        standard_probabilities = probabilities.index_select(1, amino_acid_ids)
        best_standard_indexes = standard_probabilities.argmax(dim=-1)
        completed = list(sequence)
        predictions = []
        for sequence_position, best_index, row_probabilities in zip(
            sequence_positions,
            best_standard_indexes.tolist(),
            standard_probabilities,
        ):
            residue = STANDARD_AMINO_ACIDS[int(best_index)]
            confidence = float(row_probabilities[int(best_index)].item())
            completed[sequence_position] = residue
            predictions.append(
                {
                    "position_1based": int(sequence_position + 1),
                    "predicted_residue": residue,
                    "confidence": confidence,
                    "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
                }
            )
        results.append(
            {
                "cache_version": ESM2_COMPLETION_CACHE_VERSION,
                "model": model_name,
                "original_sequence": sequence,
                "completed_sequence": "".join(completed),
                "predictions": predictions,
            }
        )
    return results


def complete_unknown_residues(
    sequences: Iterable[str],
    cache_dir: str,
    model_name: str = ESM2_COMPLETION_MODEL,
    max_batch_size: int = 8,
    max_padded_residues: int = 1200,
    logger: Callable[[str], None] = print,
) -> tuple[dict[str, str], list[dict]]:
    """Replace X residues with ESM-2 predictions and return an audit trail."""
    unique_sequences = list(dict.fromkeys(str(item) for item in sequences))
    completion_map = {sequence: sequence for sequence in unique_sequences}
    affected = [sequence for sequence in unique_sequences if "X" in sequence]
    if not affected:
        return completion_map, []

    total_x = sum(sequence.count("X") for sequence in affected)
    logger(
        "Unknown-residue completion: "
        f"{len(affected)} unique sequence(s), {total_x} X position(s)."
    )
    high_x_burden = {}
    for sequence_index, sequence in enumerate(affected, start=1):
        x_count = sequence.count("X")
        is_high = (
            x_count > HIGH_X_COUNT_THRESHOLD
            or x_count / len(sequence) > HIGH_X_RATIO_THRESHOLD
        )
        high_x_burden[sequence] = is_high
        if is_high:
            logger(
                f"Warning: sequence {sequence_index} contains {x_count} X "
                f"residue(s) ({x_count / len(sequence):.1%}). Treat its "
                "completed sequence with caution."
            )

    payloads = []
    uncached = []
    for sequence in affected:
        cached = _load_cached_completion(sequence, cache_dir, model_name)
        if cached is None:
            uncached.append(sequence)
        else:
            payloads.append(cached)
    if payloads:
        logger(
            "Unknown-residue completion cache: "
            f"reused {len(payloads)} sequence(s)."
        )

    if uncached:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger(
            f"Loading {model_name} for X completion on {device}. "
            "The model is loaded only for sequences containing X."
        )
        if device == "cpu":
            logger("Warning: ESM-2 650M X completion is much faster on a GPU.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(Path(cache_dir) / "huggingface"),
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            cache_dir=str(Path(cache_dir) / "huggingface"),
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        effective_batch_size = max_batch_size if device == "cuda" else 1
        effective_residue_limit = max_padded_residues if device == "cuda" else 400
        batches = _build_batches(
            uncached,
            max_batch_size=effective_batch_size,
            max_padded_residues=effective_residue_limit,
        )
        try:
            for batch_index, batch in enumerate(batches, start=1):
                logger(
                    f"Completing X batch {batch_index}/{len(batches)}: "
                    f"{len(batch)} sequence(s), "
                    f"{sum(item.count('X') for item in batch)} position(s), "
                    f"maximum length {max(map(len, batch))}."
                )
                batch_payloads = _predict_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    model_name,
                )
                for payload in batch_payloads:
                    _write_cached_completion(payload, cache_dir, model_name)
                payloads.extend(batch_payloads)
        finally:
            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
        logger("ESM-2 X-completion model released from memory.")

    payload_by_sequence = {
        payload["original_sequence"]: payload for payload in payloads
    }
    audit_rows = []
    low_confidence = 0
    logged_predictions = 0
    for sequence_index, sequence in enumerate(affected, start=1):
        payload = payload_by_sequence[sequence]
        completed_sequence = payload["completed_sequence"]
        completion_map[sequence] = completed_sequence
        for prediction in payload["predictions"]:
            prediction = dict(prediction)
            low_confidence += int(prediction["low_confidence"])
            audit_rows.append(
                {
                    "sequence_index": sequence_index,
                    "position_1based": prediction["position_1based"],
                    "predicted_residue": prediction["predicted_residue"],
                    "confidence": prediction["confidence"],
                    "low_confidence": prediction["low_confidence"],
                    "high_x_burden": high_x_burden[sequence],
                    "model": model_name,
                    "original_sequence": sequence,
                    "completed_sequence": completed_sequence,
                }
            )
            if logged_predictions < MAX_LOGGED_PREDICTIONS:
                logger(
                    f"X completion: sequence {sequence_index}, position "
                    f"{prediction['position_1based']} -> "
                    f"{prediction['predicted_residue']} "
                    f"(confidence {prediction['confidence']:.3f})."
                )
                logged_predictions += 1

    if len(audit_rows) > MAX_LOGGED_PREDICTIONS:
        logger(
            f"X-completion log preview is limited to "
            f"{MAX_LOGGED_PREDICTIONS} positions; the audit report contains "
            "all predictions."
        )

    logger(
        "Unknown-residue completion finished: "
        f"{len(affected)} sequence(s), {len(audit_rows)} position(s), "
        f"{low_confidence} below confidence {LOW_CONFIDENCE_THRESHOLD:.2f}."
    )
    return completion_map, audit_rows
