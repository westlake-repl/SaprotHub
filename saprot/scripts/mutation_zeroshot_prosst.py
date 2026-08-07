import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from saprot.model.prosst.specs import resolve_structure_vocab_size
except ImportError:
    from data.pdb2prosst import (
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from model.prosst.specs import resolve_structure_vocab_size


MUTATION_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


def parse_mutation(mutant: str, sequence: str) -> List[Tuple[str, int, str]]:
    parsed = []
    for item in str(mutant).split(":"):
        item = item.strip().upper()
        match = MUTATION_RE.match(item)
        if match is None:
            raise ValueError(
                f"Invalid mutation '{item}'. Expected format like H87Y or H87Y:V162M."
            )

        wt, pos, mt = match.groups()
        idx = int(pos) - 1
        if idx < 0 or idx >= len(sequence):
            raise ValueError(
                f"Mutation '{item}' position is out of range for sequence length {len(sequence)}."
            )
        if sequence[idx] != wt:
            raise ValueError(
                f"Mutation '{item}' WT amino acid mismatch: sequence[{idx + 1}] is "
                f"'{sequence[idx]}', not '{wt}'."
            )
        parsed.append((wt, idx, mt))

    return parsed


def _row_structure_entry(
    row: pd.Series,
    csv_dir: Path,
    structure_base_dir: str = None,
) -> Dict:
    entry = {}
    if "structure_tokens" in row.index and pd.notna(row["structure_tokens"]) and str(row["structure_tokens"]).strip():
        entry["structure_tokens"] = row["structure_tokens"]
    else:
        path_column = None
        for column in ["structure_path", "pdb_path"]:
            if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
                path_column = column
                break
        if path_column is None:
            raise ValueError(
                "Each ProSST mutation row needs structure_tokens, "
                "structure_path, or pdb_path."
            )

        pdb_path = Path(str(row[path_column]).strip())
        if not pdb_path.is_absolute():
            candidates = [csv_dir / pdb_path]
            if structure_base_dir is not None and str(structure_base_dir).strip():
                candidates.append(Path(structure_base_dir) / pdb_path)
            pdb_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        entry["pdb_path"] = str(pdb_path)
        if "chain_id" in row.index and pd.notna(row["chain_id"]):
            entry["chain_id"] = str(row["chain_id"]).strip()
        elif "chain" in row.index and pd.notna(row["chain"]):
            entry["chain_id"] = str(row["chain"]).strip()

    if (
        "structure_vocab_size" in row.index
        and pd.notna(row["structure_vocab_size"])
        and str(row["structure_vocab_size"]).strip()
    ):
        entry["structure_vocab_size"] = row["structure_vocab_size"]

    return entry


def _prepare_inputs(
    tokenizer,
    sequence: str,
    structure_tokens: Sequence[int],
    structure_vocab_size: int,
    device,
):
    tokenized = tokenizer([sequence], return_tensors="pt")
    encoded_ss = encode_structure_tokens(
        structure_tokens,
        structure_vocab_size=structure_vocab_size,
    )
    if tokenized["input_ids"].shape[1] != len(encoded_ss):
        raise ValueError(
            "ProSST tokenizer input length and structure token length differ after "
            f"adding special tokens: input_ids={tokenized['input_ids'].shape[1]}, "
            f"ss_input_ids={len(encoded_ss)}."
        )

    inputs = {key: value.to(device) for key, value in tokenized.items()}
    inputs["ss_input_ids"] = torch.tensor([encoded_ss], dtype=torch.long, device=device)
    return inputs


@torch.no_grad()
def score_sequence(
    model,
    tokenizer,
    sequence: str,
    structure_tokens: Sequence[int],
    structure_vocab_size: int,
    device,
):
    inputs = _prepare_inputs(
        tokenizer,
        sequence,
        structure_tokens,
        structure_vocab_size,
        device,
    )
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        ss_input_ids=inputs["ss_input_ids"],
        return_dict=True,
    )
    logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)
    if logits.shape[1] != len(sequence):
        raise ValueError(
            f"Model residue logits length {logits.shape[1]} does not match sequence length {len(sequence)}."
        )
    return logits[0]


def score_mutants(
    input_csv: str,
    output_csv: str,
    model_path: str = "AI4Protein/ProSST-2048",
    cache_dir: str = None,
    structure_vocab_size: Optional[int] = None,
    device: str = None,
    structure_base_dir: str = None,
) -> pd.DataFrame:
    structure_vocab_size = resolve_structure_vocab_size(
        model_path,
        structure_vocab_size,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.lower()

    required = {"sequence", "mutant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ProSST mutation CSV missing columns: {sorted(missing)}")
    if (
        "structure_tokens" not in df.columns
        and "structure_path" not in df.columns
        and "pdb_path" not in df.columns
    ):
        raise ValueError(
            "ProSST mutation CSV requires structure_tokens, structure_path, or pdb_path."
        )
    if df.empty:
        raise ValueError("ProSST mutation CSV contains no rows.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    vocab = tokenizer.get_vocab()
    csv_dir = Path(input_csv).resolve().parent
    logit_cache = {}
    scores = []

    for row_idx, row in df.iterrows():
        sequence = str(row["sequence"]).strip().upper()
        if not sequence or sequence == "NAN":
            raise ValueError(f"ProSST mutation row {row_idx} has an empty sequence.")
        entry = _row_structure_entry(
            row,
            csv_dir,
            structure_base_dir=structure_base_dir,
        )
        structure_tokens = get_structure_tokens_from_entry(
            entry,
            cache_dir=cache_dir,
            structure_vocab_size=structure_vocab_size,
        )
        validate_sequence_and_structure(sequence, structure_tokens, context=f"row {row_idx}")

        cache_key = (sequence, tuple(structure_tokens))
        if cache_key not in logit_cache:
            logit_cache[cache_key] = score_sequence(
                model,
                tokenizer,
                sequence,
                structure_tokens,
                structure_vocab_size,
                device,
            )
        logits = logit_cache[cache_key]

        score = 0.0
        for wt, idx, mt in parse_mutation(row["mutant"], sequence):
            if wt not in vocab or mt not in vocab:
                raise ValueError(f"Mutation uses amino acid outside ProSST vocab: {wt}->{mt}")
            score += (logits[idx, vocab[mt]] - logits[idx, vocab[wt]]).item()
        scores.append(score)

    df["score"] = scores
    df.to_csv(output_csv, index=False)
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure_base_dir", default=None)
    return parser.parse_args()


def main():
    args = get_args()
    score_mutants(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        device=args.device,
        structure_base_dir=args.structure_base_dir,
    )


if __name__ == "__main__":
    main()
