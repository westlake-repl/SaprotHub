import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer

_SAPROT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SAPROT_DIR.parent
for _path in [str(_REPO_ROOT), str(_SAPROT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from saprot.data.pdb2prosst import (
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from saprot.data.prosst_inputs import (
        find_sequence_column,
        prepare_prosst_batch,
        structure_entry_from_row,
    )
    from saprot.model.prosst.prosst_classification_model import ProSSTClassificationModel
    from saprot.model.prosst.prosst_regression_model import ProSSTRegressionModel
    from saprot.model.prosst.prosst_pair_classification_model import (
        ProSSTPairClassificationModel,
    )
    from saprot.model.prosst.prosst_pair_regression_model import (
        ProSSTPairRegressionModel,
    )
    from saprot.model.prosst.prosst_token_classification_model import (
        ProSSTTokenClassificationModel,
    )
    from saprot.model.prosst.specs import resolve_structure_vocab_size
except ImportError:
    from data.pdb2prosst import (
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from data.prosst_inputs import (
        find_sequence_column,
        prepare_prosst_batch,
        structure_entry_from_row,
    )
    from model.prosst.prosst_classification_model import ProSSTClassificationModel
    from model.prosst.prosst_regression_model import ProSSTRegressionModel
    from model.prosst.prosst_pair_classification_model import (
        ProSSTPairClassificationModel,
    )
    from model.prosst.prosst_pair_regression_model import (
        ProSSTPairRegressionModel,
    )
    from model.prosst.prosst_token_classification_model import (
        ProSSTTokenClassificationModel,
    )
    from model.prosst.specs import resolve_structure_vocab_size


PAIR_TASK_TYPES = {"pair_classification", "pair_regression"}
CLASSIFICATION_TASK_TYPES = {
    "classification",
    "token_classification",
    "pair_classification",
}
SUPPORTED_TASK_TYPES = {
    *CLASSIFICATION_TASK_TYPES,
    "regression",
    "pair_regression",
}
LORA_METADATA_FILENAME = "colabprosst.json"


def load_prosst_downstream_model(
    task_type: str,
    model_path: str,
    checkpoint_path: str,
    num_labels: int,
    structure_vocab_size: int,
    device: torch.device,
    load_pretrained: bool = False,
):
    checkpoint = Path(checkpoint_path)
    is_lora_adapter = checkpoint.is_dir()
    if is_lora_adapter and not (checkpoint / "adapter_config.json").is_file():
        raise ValueError(
            f"LoRA checkpoint directory has no adapter_config.json: {checkpoint}"
        )
    common_kwargs = {
        "config_path": model_path,
        "structure_vocab_size": structure_vocab_size,
        "load_pretrained": True if is_lora_adapter else load_pretrained,
        "lr_scheduler_kwargs": {
            "class": "ConstantLRScheduler",
            "init_lr": 0.0,
        },
        "optimizer_kwargs": {
            "class": "AdamW",
            "betas": [0.9, 0.98],
            "weight_decay": 0.01,
        },
    }
    if is_lora_adapter:
        common_kwargs["lora_kwargs"] = {
            "is_trainable": False,
            "num_lora": 1,
            "config_list": [
                {"lora_config_path": str(checkpoint)}
            ],
        }
    else:
        common_kwargs["from_checkpoint"] = checkpoint_path
    if task_type == "classification":
        model = ProSSTClassificationModel(
            num_labels=num_labels,
            **common_kwargs,
        )
    elif task_type == "token_classification":
        model = ProSSTTokenClassificationModel(
            num_labels=num_labels,
            **common_kwargs,
        )
    elif task_type == "pair_classification":
        model = ProSSTPairClassificationModel(
            num_labels=num_labels,
            **common_kwargs,
        )
    elif task_type == "regression":
        model = ProSSTRegressionModel(**common_kwargs)
    elif task_type == "pair_regression":
        model = ProSSTPairRegressionModel(**common_kwargs)
    else:
        raise ValueError(f"Unsupported ProSST prediction task_type: {task_type}.")

    model = model.to(device)
    model.eval()
    return model


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    task_type: str,
    model_path: str,
    structure_vocab_size: int,
    num_labels: Optional[int] = None,
) -> None:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        metadata_path = checkpoint / LORA_METADATA_FILENAME
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
    else:
        try:
            checkpoint_state = torch.load(checkpoint, map_location="cpu")
        except Exception:
            # The regular model loader will report unreadable checkpoints.
            return
        if not isinstance(checkpoint_state, dict):
            return
        metadata = checkpoint_state.get("colabprosst")
    if not isinstance(metadata, dict):
        return

    expected = {
        "task": task_type,
        "base_model": model_path,
        "structure_vocab_size": int(structure_vocab_size),
    }
    if (
        task_type in CLASSIFICATION_TASK_TYPES
        and num_labels is not None
    ):
        expected["num_labels"] = int(num_labels)
    mismatches = [
        f"{key}={metadata[key]!r} (checkpoint), expected {value!r}"
        for key, value in expected.items()
        if key in metadata and metadata[key] != value
    ]
    if mismatches:
        raise ValueError(
            "The ProSST checkpoint is incompatible with the selected settings: "
            + "; ".join(mismatches)
            + ". Select the checkpoint's original base model and task."
        )


@torch.no_grad()
def predict_csv(
    input_csv: str,
    output_csv: str,
    task_type: str,
    checkpoint_path: str,
    model_path: str = "AI4Protein/ProSST-2048",
    num_labels: int = 2,
    batch_size: int = 1,
    cache_dir: str = None,
    structure_vocab_size: Optional[int] = None,
    max_length: int = 2046,
    device: str = None,
    structure_base_dir: str = None,
    load_pretrained: bool = False,
) -> pd.DataFrame:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported ProSST prediction task_type: {task_type}."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if task_type in CLASSIFICATION_TASK_TYPES and num_labels < 2:
        raise ValueError("Classification num_labels must be at least 2.")
    structure_vocab_size = resolve_structure_vocab_size(
        model_path,
        structure_vocab_size,
    )
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for ProSST prediction.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"ProSST checkpoint does not exist: {checkpoint}")
    validate_checkpoint_compatibility(
        str(checkpoint),
        task_type,
        model_path,
        structure_vocab_size,
        num_labels,
    )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("ProSST prediction CSV contains no rows.")
    is_pair_task = task_type in PAIR_TASK_TYPES
    if is_pair_task:
        lower_columns = {column.lower(): column for column in df.columns}
        missing = [
            column
            for column in ["sequence_1", "sequence_2"]
            if column not in lower_columns
        ]
        if missing:
            raise ValueError(
                f"ProSST pair prediction CSV missing columns: {missing}."
            )
        sequence_columns = [
            lower_columns["sequence_1"],
            lower_columns["sequence_2"],
        ]
    else:
        sequence_columns = [find_sequence_column(df.columns)]
    csv_dir = Path(input_csv).resolve().parent

    sequence_groups: List[List[str]] = [[] for _column in sequence_columns]
    structure_groups: List[List[List[int]]] = [
        [] for _column in sequence_columns
    ]
    for row_idx, row in df.iterrows():
        for group_idx, sequence_column in enumerate(sequence_columns):
            sequence = str(row[sequence_column]).strip().upper()
            pair_index = group_idx + 1 if is_pair_task else None
            if not sequence or sequence == "NAN":
                subject = f" pair protein {pair_index}" if pair_index else ""
                raise ValueError(
                    f"ProSST prediction row {row_idx}{subject} has an empty sequence."
                )
            entry = structure_entry_from_row(
                row,
                csv_dir,
                structure_base_dir=structure_base_dir,
                pair_index=pair_index,
            )
            structure_tokens = get_structure_tokens_from_entry(
                entry,
                cache_dir=cache_dir,
                structure_vocab_size=structure_vocab_size,
            )
            context = f"row {row_idx}"
            if pair_index is not None:
                context += f" pair protein {pair_index}"
            validate_sequence_and_structure(
                sequence,
                structure_tokens,
                context=context,
            )
            sequence_groups[group_idx].append(sequence)
            structure_groups[group_idx].append(
                [int(token) for token in structure_tokens]
            )

    sequences = sequence_groups[0]
    structure_tokens_list = structure_groups[0]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_prosst_downstream_model(
        task_type,
        model_path,
        checkpoint_path,
        num_labels,
        structure_vocab_size,
        device,
        load_pretrained=load_pretrained,
    )

    output_chunks = []
    token_predictions = []
    for start in range(0, len(sequences), batch_size):
        stop = start + batch_size
        batch_inputs = prepare_prosst_batch(
            tokenizer,
            sequences[start:stop],
            structure_tokens_list[start:stop],
            max_length=max_length,
            structure_vocab_size=structure_vocab_size,
            device=device,
        )
        if is_pair_task:
            batch_inputs_2 = prepare_prosst_batch(
                tokenizer,
                sequence_groups[1][start:stop],
                structure_groups[1][start:stop],
                max_length=max_length,
                structure_vocab_size=structure_vocab_size,
                device=device,
            )
            logits = model.forward(batch_inputs, batch_inputs_2).detach().cpu()
        else:
            logits = model.forward(batch_inputs).detach().cpu()
        if task_type == "token_classification":
            for row_idx, sequence in enumerate(sequences[start:stop]):
                residue_count = min(len(sequence), max_length)
                encoded_count = int(
                    batch_inputs["attention_mask"][row_idx].sum().item()
                )
                if encoded_count != residue_count + 2:
                    raise ValueError(
                        "ProSST tokenizer must produce one token per residue plus "
                        "CLS/EOS for residue-level prediction: "
                        f"encoded={encoded_count}, expected={residue_count + 2}."
                    )
                residue_logits = logits[row_idx, 1 : 1 + residue_count]
                if residue_logits.shape[0] != residue_count:
                    raise ValueError(
                        "Token-classification output does not align with the "
                        f"input sequence: logits={residue_logits.shape[0]}, "
                        f"residues={residue_count}."
                    )
                probabilities = torch.softmax(residue_logits, dim=-1)
                token_predictions.append(
                    (
                        probabilities.argmax(dim=-1),
                        probabilities.max(dim=-1).values,
                        probabilities,
                    )
                )
        else:
            output_chunks.append(logits)

    result = df.copy()
    if task_type == "token_classification":
        result["prediction_length"] = [
            len(predictions)
            for predictions, _confidence, _probabilities in token_predictions
        ]
        result["predicted_labels"] = [
            " ".join(str(value.item()) for value in predictions)
            for predictions, _confidence, _probabilities in token_predictions
        ]
        result["confidence"] = [
            " ".join(f"{value.item():.8g}" for value in confidence)
            for _predictions, confidence, _probabilities in token_predictions
        ]
        for label_idx in range(num_labels):
            result[f"prob_{label_idx}"] = [
                " ".join(
                    f"{value.item():.8g}"
                    for value in probabilities[:, label_idx]
                )
                for _predictions, _confidence, probabilities in token_predictions
            ]
    else:
        outputs = torch.cat(output_chunks, dim=0)
    if task_type in {"classification", "pair_classification"}:
        probabilities = torch.softmax(outputs, dim=-1)
        result["pred"] = probabilities.argmax(dim=-1).numpy()
        for idx in range(probabilities.shape[-1]):
            result[f"prob_{idx}"] = probabilities[:, idx].numpy()
    elif task_type in {"regression", "pair_regression"}:
        result["pred"] = outputs.reshape(-1).numpy()

    result.to_csv(output_csv, index=False)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--task_type",
        required=True,
        choices=sorted(SUPPORTED_TASK_TYPES),
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2046)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure_base_dir", default=None)
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help=(
            "Load the full base ProSST weights before applying the checkpoint. "
            "By default prediction builds the model from config because "
            "ColabProSST checkpoints contain the full model state dict."
        ),
    )
    return parser.parse_args()


def main():
    args = get_args()
    predict_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        task_type=args.task_type,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        max_length=args.max_length,
        device=args.device,
        structure_base_dir=args.structure_base_dir,
        load_pretrained=args.load_pretrained,
    )


if __name__ == "__main__":
    main()
