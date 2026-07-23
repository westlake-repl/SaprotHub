import gc
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


SS_PAD_TOKEN_ID = 0
SS_CLS_TOKEN_ID = 1
SS_EOS_TOKEN_ID = 2
SS_TOKEN_OFFSET = 3
SUPPORTED_STRUCTURE_VOCAB_SIZES = {20, 64, 128, 512, 1024, 2048, 4096}

_PREDICTOR_CACHE = {}


class ProSSTStructureError(RuntimeError):
    """Raised when ProSST structure token generation fails."""


def _patch_threadpoolctl_stale_library_scan() -> None:
    """Ignore native libraries removed by an in-process package replacement."""
    try:
        import threadpoolctl
    except ImportError:
        return

    controller_class = getattr(threadpoolctl, "ThreadpoolController", None)
    if (
        controller_class is None
        or not hasattr(controller_class, "_make_controller_from_path")
        or getattr(controller_class, "_colabprosst_stale_library_guard", False)
    ):
        return

    original = controller_class._make_controller_from_path

    def make_controller_from_path(controller, filepath):
        try:
            return original(controller, filepath)
        except OSError:
            if filepath and not os.path.exists(os.fsdecode(filepath)):
                return None
            raise

    controller_class._make_controller_from_path = make_controller_from_path
    controller_class._colabprosst_stale_library_guard = True


def _add_prosst_to_path() -> None:
    candidates = []
    env_home = os.environ.get("PROSST_HOME")
    if env_home:
        candidates.append(Path(env_home))

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2].parent / "ProSST",  # ../ProSST beside SaprotHub
            Path.cwd() / "ProSST",
            Path("/content/ProSST"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _load_sst_predictor():
    _add_prosst_to_path()
    _patch_threadpoolctl_stale_library_scan()
    try:
        sst_module = importlib.import_module("prosst.structure.get_sst_seq")
    except Exception as exc:
        raise ImportError(
            "Cannot import ProSST SSTPredictor. Install/clone the official ProSST "
            "repository and set PROSST_HOME, or place ProSST beside SaprotHub."
        ) from exc

    _patch_sst_module(sst_module)
    return sst_module.SSTPredictor


def _patch_sst_module(sst_module) -> None:
    if getattr(sst_module, "_colabprosst_patched", False):
        return

    original_iter_parallel_map = sst_module.iter_parallel_map

    def iter_parallel_map(func, data, workers: int = 2):
        if workers <= 0:
            return map(func, data)
        return original_iter_parallel_map(func, data, workers)

    sst_module.iter_parallel_map = iter_parallel_map
    sst_module._colabprosst_patched = True


def get_sst_predictor(
    structure_vocab_size: int = 2048,
    device: Optional[str] = None,
    **predictor_kwargs,
):
    if structure_vocab_size not in SUPPORTED_STRUCTURE_VOCAB_SIZES:
        raise ValueError(
            f"Unsupported ProSST structure_vocab_size={structure_vocab_size}. "
            f"Expected one of {sorted(SUPPORTED_STRUCTURE_VOCAB_SIZES)}."
        )

    predictor_kwargs.setdefault("num_processes", 0)
    cache_key = (
        structure_vocab_size,
        device,
        tuple(sorted((key, repr(value)) for key, value in predictor_kwargs.items())),
    )
    if cache_key not in _PREDICTOR_CACHE:
        SSTPredictor = _load_sst_predictor()
        _PREDICTOR_CACHE[cache_key] = SSTPredictor(
            structure_vocab_size=structure_vocab_size,
            device=device,
            **predictor_kwargs,
        )

    return _PREDICTOR_CACHE[cache_key]


def clear_sst_predictor_cache() -> int:
    """Release cached ProSST structure quantizers before downstream modeling."""
    released = len(_PREDICTOR_CACHE)
    _PREDICTOR_CACHE.clear()
    gc.collect()
    try:
        import torch
    except ImportError:
        return released
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released


def _parse_structure_file(structure_path: Union[str, Path]):
    try:
        from Bio import PDB
    except Exception as exc:
        raise ImportError("Biopython is required for ProSST structure parsing.") from exc

    structure_path = Path(structure_path)
    suffix = structure_path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        parser = PDB.PDBParser(QUIET=True)
    elif suffix in {".cif", ".mmcif"}:
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        raise NotImplementedError(
            "ColabProSST supports PDB and mmCIF files for ProSST quantization. "
            f"Unsupported structure format: {suffix or '<none>'}."
        )

    return parser.get_structure("protein", str(structure_path))


def _write_structure_to_temp_pdb(
    structure_path: Union[str, Path],
    chain_id: Optional[str] = None,
) -> str:
    try:
        from Bio import PDB
    except Exception as exc:
        raise ImportError("Biopython is required for ProSST structure conversion.") from exc

    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):  # noqa: N802 - Biopython API
            return chain_id is None or chain.id == chain_id

    structure = _parse_structure_file(structure_path)
    chain_ids = [chain.id for model in structure for chain in model]
    if chain_id is not None and chain_id not in chain_ids:
        raise ValueError(
            f"Chain '{chain_id}' not found in {structure_path}. "
            f"Available chains: {chain_ids}"
        )

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    handle.close()
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(handle.name, ChainSelect())
    return handle.name


def _prepare_quantize_path(
    structure_path: Union[str, Path],
    chain_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    structure_path = Path(structure_path)
    suffix = structure_path.suffix.lower()
    if suffix in {".pdb", ".ent"} and chain_id is None:
        return str(structure_path), None

    temp_pdb = _write_structure_to_temp_pdb(structure_path, chain_id=chain_id)
    return temp_pdb, temp_pdb


def parse_structure_tokens(value: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(value, (list, tuple)):
        tokens = [int(item) for item in value]
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("structure_tokens is empty.")

        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("JSON structure_tokens must be a list of integers.")
            tokens = [int(item) for item in parsed]
        else:
            normalized = value.replace(",", " ")
            tokens = [int(item) for item in normalized.split()]
    else:
        raise TypeError(
            "structure_tokens must be a list/tuple of integers or a comma/space "
            f"separated string, got {type(value).__name__}."
        )

    if len(tokens) == 0:
        raise ValueError("structure_tokens is empty.")

    return tokens


def serialize_structure_tokens(tokens: Sequence[int]) -> str:
    return " ".join(str(int(token)) for token in tokens)


def validate_structure_vocab_metadata(
    entry: Dict[str, Any],
    structure_vocab_size: int,
) -> None:
    value = entry.get("structure_vocab_size")
    if value is None or str(value).strip() == "":
        return

    try:
        numeric_value = float(value)
        actual_size = int(numeric_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid structure_vocab_size metadata: {value!r}."
        ) from exc
    if numeric_value != actual_size:
        raise ValueError(f"Invalid structure_vocab_size metadata: {value!r}.")
    if actual_size != int(structure_vocab_size):
        raise ValueError(
            "ProSST structure token vocabulary mismatch: CSV tokens use "
            f"structure_vocab_size={actual_size}, but the selected model requires "
            f"{structure_vocab_size}."
        )


def encode_structure_tokens(
    structure_tokens: Union[str, Sequence[int]],
    structure_vocab_size: int = 2048,
    add_special_tokens: bool = True,
) -> List[int]:
    raw_tokens = parse_structure_tokens(structure_tokens)
    bad_tokens = [
        token for token in raw_tokens if token < 0 or token >= structure_vocab_size
    ]
    if bad_tokens:
        example = bad_tokens[:5]
        raise ValueError(
            f"ProSST raw structure tokens must be in [0, {structure_vocab_size - 1}], "
            f"found {example}."
        )

    encoded = [token + SS_TOKEN_OFFSET for token in raw_tokens]
    if add_special_tokens:
        encoded = [SS_CLS_TOKEN_ID, *encoded, SS_EOS_TOKEN_ID]

    return encoded


def pad_structure_input_ids(
    encoded_structure_ids: Sequence[Sequence[int]],
    target_length: int,
    pad_token_id: int = SS_PAD_TOKEN_ID,
) -> List[List[int]]:
    padded = []
    for ids in encoded_structure_ids:
        ids = list(ids)
        if len(ids) > target_length:
            raise ValueError(
                f"Encoded ProSST structure length {len(ids)} exceeds tokenizer "
                f"input length {target_length}."
            )
        padded.append(ids + [pad_token_id] * (target_length - len(ids)))

    return padded


def _cache_path(
    pdb_path: Union[str, Path],
    cache_dir: Union[str, Path],
    structure_vocab_size: int,
    chain_id: Optional[str],
) -> Path:
    path = Path(pdb_path).resolve()
    stat = path.stat()
    digest = hashlib.sha1(
        "|".join(
            [
                str(path),
                str(stat.st_mtime_ns),
                str(stat.st_size),
                str(structure_vocab_size),
                chain_id or "",
            ]
        ).encode()
    ).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def quantize_structure(
    pdb_path: Union[str, Path],
    structure_vocab_size: int = 2048,
    chain_id: Optional[str] = None,
    return_residue_seq: bool = True,
    device: Optional[str] = None,
    error_file: Optional[Union[str, Path]] = None,
    cache_subgraph_dir: Optional[Union[str, Path]] = None,
    **predictor_kwargs,
) -> Union[List[int], Dict[str, Any]]:
    structure_path = Path(pdb_path)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file does not exist: {structure_path}")

    suffix = structure_path.suffix.lower()
    if suffix not in {".pdb", ".ent", ".cif", ".mmcif"}:
        raise NotImplementedError(
            "ColabProSST supports PDB and mmCIF files for ProSST quantization. "
            f"Unsupported structure format: {suffix or '<none>'}."
        )

    predictor = get_sst_predictor(
        structure_vocab_size=structure_vocab_size,
        device=device,
        **predictor_kwargs,
    )

    temp_pdb = None
    try:
        quantize_path, temp_pdb = _prepare_quantize_path(structure_path, chain_id)

        try:
            results = predictor.predict_from_pdb(
                quantize_path,
                error_file=str(error_file) if error_file is not None else None,
                cache_subgraph_dir=(
                    str(cache_subgraph_dir) if cache_subgraph_dir is not None else None
                ),
            )
        except Exception as exc:
            raise ProSSTStructureError(
                f"Failed to quantize structure with ProSST: {structure_path}"
            ) from exc

        if not results:
            raise ProSSTStructureError(
                f"ProSST quantization returned no result for {structure_path}."
            )

        result = results[0]
        token_key = f"{structure_vocab_size}_sst_seq"
        if token_key not in result:
            matching_keys = [key for key in result if str(key).endswith(token_key)]
            if len(matching_keys) == 1:
                token_key = matching_keys[0]
            else:
                raise ProSSTStructureError(
                    f"ProSST quantization output does not contain '{token_key}'. "
                    f"Available keys: {list(result.keys())}"
                )

        sequence = result.get("aa_seq")
        structure_tokens = [int(token) for token in result[token_key]]
        if sequence is not None and len(sequence) != len(structure_tokens):
            raise ProSSTStructureError(
                "ProSST structure token length does not match parsed residue "
                f"sequence length: tokens={len(structure_tokens)}, sequence={len(sequence)}."
            )

        if return_residue_seq:
            return {
                "sequence": sequence,
                "structure_tokens": structure_tokens,
                "pdb_path": str(structure_path),
                "chain_id": chain_id,
                "structure_vocab_size": structure_vocab_size,
            }

        return structure_tokens
    finally:
        if temp_pdb is not None and os.path.exists(temp_pdb):
            os.remove(temp_pdb)


def load_or_quantize_structure(
    pdb_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    structure_vocab_size: int = 2048,
    chain_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if cache_dir is None:
        return quantize_structure(
            pdb_path,
            structure_vocab_size=structure_vocab_size,
            chain_id=chain_id,
            return_residue_seq=True,
            **kwargs,
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(pdb_path, cache_dir, structure_vocab_size, chain_id)

    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    result = quantize_structure(
        pdb_path,
        structure_vocab_size=structure_vocab_size,
        chain_id=chain_id,
        return_residue_seq=True,
        **kwargs,
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)

    return result


def get_structure_tokens_from_entry(
    entry: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    structure_vocab_size: int = 2048,
) -> List[int]:
    validate_structure_vocab_metadata(entry, structure_vocab_size)
    tokens = entry.get("structure_tokens")
    if tokens is not None and str(tokens).strip() != "":
        return parse_structure_tokens(tokens)

    pdb_path = entry.get("pdb_path")
    if pdb_path is None or str(pdb_path).strip() == "":
        raise ValueError("Either structure_tokens or pdb_path is required for ProSST.")

    result = load_or_quantize_structure(
        pdb_path,
        cache_dir=cache_dir,
        structure_vocab_size=structure_vocab_size,
        chain_id=entry.get("chain_id"),
    )
    return [int(token) for token in result["structure_tokens"]]


def validate_sequence_and_structure(
    sequence: str,
    structure_tokens: Sequence[int],
    context: str = "sample",
) -> None:
    if len(sequence) != len(structure_tokens):
        raise ValueError(
            f"ProSST {context} has mismatched sequence and structure token lengths: "
            f"sequence={len(sequence)}, structure_tokens={len(structure_tokens)}."
        )


def ensure_int_tokens(tokens: Iterable[int]) -> List[int]:
    return [int(token) for token in tokens]
