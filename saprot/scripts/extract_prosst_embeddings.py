import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    from saprot.model.prosst.specs import resolve_structure_vocab_size
    from saprot.scripts.predict_prosst import (
        load_prosst_downstream_model,
        validate_adapter_compatibility,
    )
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
    from model.prosst.specs import resolve_structure_vocab_size
    from scripts.predict_prosst import (
        load_prosst_downstream_model,
        validate_adapter_compatibility,
    )


EMBEDDING_LEVELS = {"protein", "residue", "both"}


@torch.no_grad()
def extract_embeddings(
    input_csv: str,
    output_pt: str,
    model_path: str = "AI4Protein/ProSST-2048",
    level: str = "protein",
    output_index_csv: str = None,
    cache_dir: str = None,
    structure_vocab_size: Optional[int] = None,
    batch_size: int = 1,
    max_length: int = 2046,
    layer_index: int = -1,
    device: str = None,
    structure_base_dir: str = None,
    adapter_path: str = "",
    adapter_task_type: str = None,
    adapter_num_labels: int = 2,
) -> dict:
    level = str(level).strip().lower()
    if level not in EMBEDDING_LEVELS:
        raise ValueError(
            f"Embedding level must be one of {sorted(EMBEDDING_LEVELS)}, "
            f"got {level!r}."
        )
    if batch_size < 1:
        raise ValueError("Embedding batch_size must be at least 1.")
    if max_length < 1:
        raise ValueError("Embedding max_length must be at least 1.")

    structure_vocab_size = resolve_structure_vocab_size(
        model_path,
        structure_vocab_size,
    )
    table = pd.read_csv(input_csv)
    if table.empty:
        raise ValueError("ProSST embedding CSV contains no rows.")
    sequence_column = find_sequence_column(table.columns)
    csv_dir = Path(input_csv).resolve().parent

    sequences: List[str] = []
    structure_tokens_list: List[List[int]] = []
    for row_index, row in table.iterrows():
        sequence = str(row[sequence_column]).strip().upper()
        if not sequence or sequence == "NAN":
            raise ValueError(
                f"ProSST embedding row {row_index} has an empty sequence."
            )
        if len(sequence) > max_length:
            raise ValueError(
                f"ProSST embedding row {row_index} has {len(sequence)} "
                f"residues, exceeding max_length={max_length}. Embeddings are "
                "not silently truncated."
            )
        entry = structure_entry_from_row(
            row,
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
            context=f"row {row_index}",
        )
        sequences.append(sequence)
        structure_tokens_list.append(
            [int(token) for token in structure_tokens]
        )

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    adapter_path = str(adapter_path or "").strip()
    if adapter_path:
        if not adapter_task_type:
            raise ValueError(
                "adapter_task_type is required when extracting embeddings "
                "from a downstream adapter."
            )
        if layer_index != -1:
            raise ValueError(
                "Fine-tuned adapter embedding extraction currently uses "
                "the final hidden layer; set layer_index=-1."
            )
        validate_adapter_compatibility(
            adapter_path,
            adapter_task_type,
            model_path,
            structure_vocab_size,
            adapter_num_labels,
        )
        model = load_prosst_downstream_model(
            task_type=adapter_task_type,
            model_path=model_path,
            adapter_path=adapter_path,
            num_labels=adapter_num_labels,
            structure_vocab_size=structure_vocab_size,
            device=target_device,
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = model.to(target_device)
        model.eval()

    keep_protein = level in {"protein", "both"}
    keep_residue = level in {"residue", "both"}
    protein_embeddings = []
    residue_embeddings = []
    resolved_layer_index = None

    for start in range(0, len(sequences), batch_size):
        stop = min(start + batch_size, len(sequences))
        batch_sequences = sequences[start:stop]
        inputs = prepare_prosst_batch(
            tokenizer,
            batch_sequences,
            structure_tokens_list[start:stop],
            max_length=max_length,
            structure_vocab_size=structure_vocab_size,
            device=target_device,
        )
        if adapter_path:
            batch_embeddings = model.get_token_representations(inputs)
            resolved_layer_index = -1
        else:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                ss_input_ids=inputs["ss_input_ids"],
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError(
                    "The selected ProSST model did not return hidden states."
                )
            try:
                batch_embeddings = hidden_states[layer_index]
            except IndexError as exc:
                raise ValueError(
                    f"layer_index={layer_index} is outside the model's "
                    f"{len(hidden_states)} available hidden-state layers."
                ) from exc
            resolved_layer_index = layer_index % len(hidden_states)

        for local_index, sequence in enumerate(batch_sequences):
            encoded_length = int(
                inputs["attention_mask"][local_index].sum().item()
            )
            expected_length = len(sequence) + 2
            if encoded_length != expected_length:
                raise ValueError(
                    "ProSST tokenizer must produce one token per residue plus "
                    "CLS/EOS for embedding extraction: "
                    f"row={start + local_index}, encoded={encoded_length}, "
                    f"expected={expected_length}."
                )
            residue_embedding = batch_embeddings[
                local_index,
                1 : 1 + len(sequence),
            ].detach().float().cpu()
            if residue_embedding.shape[0] != len(sequence):
                raise ValueError(
                    "ProSST residue embedding length does not match the input "
                    f"sequence for row {start + local_index}."
                )
            if keep_protein:
                protein_embeddings.append(residue_embedding.mean(dim=0))
            if keep_residue:
                residue_embeddings.append(residue_embedding)

    bundle = {
        "format_version": 1,
        "model_path": model_path,
        "structure_vocab_size": int(structure_vocab_size),
        "embedding_level": level,
        "layer_index": int(resolved_layer_index),
        "hidden_size": int(batch_embeddings.shape[-1]),
        "dtype": "float32",
        "adapter_path": adapter_path,
        "adapter_task_type": adapter_task_type if adapter_path else None,
        "sequences": sequences,
        "sequence_lengths": torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
        ),
    }
    if keep_protein:
        bundle["protein_embeddings"] = torch.stack(protein_embeddings)
    if keep_residue:
        bundle["residue_embeddings"] = residue_embeddings

    output_path = Path(output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    index_path = (
        Path(output_index_csv)
        if output_index_csv
        else output_path.with_name(f"{output_path.stem}_index.csv")
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "embedding_index": range(len(sequences)),
            "sequence": sequences,
            "sequence_length": [len(sequence) for sequence in sequences],
        }
    ).to_csv(index_path, index=False)
    bundle["output_pt"] = str(output_path)
    bundle["output_index_csv"] = str(index_path)
    return bundle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    parser.add_argument(
        "--level",
        choices=sorted(EMBEDDING_LEVELS),
        default="protein",
    )
    parser.add_argument("--output_index_csv", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2046)
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure_base_dir", default=None)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--adapter_task_type", default=None)
    parser.add_argument("--adapter_num_labels", type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    bundle = extract_embeddings(
        input_csv=args.input_csv,
        output_pt=args.output_pt,
        model_path=args.model_path,
        level=args.level,
        output_index_csv=args.output_index_csv,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer_index=args.layer_index,
        device=args.device,
        structure_base_dir=args.structure_base_dir,
        adapter_path=args.adapter_path,
        adapter_task_type=args.adapter_task_type,
        adapter_num_labels=args.adapter_num_labels,
    )
    print("saved embeddings:", bundle["output_pt"])
    print("saved embedding index:", bundle["output_index_csv"])


if __name__ == "__main__":
    main()
