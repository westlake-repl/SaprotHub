import os
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - handled gracefully
    AutoTokenizer = None  # type: ignore

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore

try:
    from esm.sdk.api import ESMProtein  # type: ignore
except ImportError:  # pragma: no cover
    ESMProtein = None  # type: ignore


def _default_cache_dir() -> Path:
    env_dir = os.environ.get("SAPROT_TOKENIZER_CACHE")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / "saprot_tokenizers"


def _should_try_fallback(model_name: Any) -> bool:
    if not isinstance(model_name, str):
        return False
    # We only care about repo identifiers. Local directories should keep
    # their original behaviour as they are expected to contain the required
    # tokenizer assets already.
    return "/" in model_name


def _is_esmc_repo(model_name: Any) -> bool:
    if not isinstance(model_name, str):
        return False
    lowered = model_name.lower()
    return "evolutionaryscale/esmc-" in lowered


def _load_esmc_tokenizer(model_name: str):
    if ESMProtein is None:
        raise EnvironmentError(
            "Loading tokenizer for EvolutionaryScale ESMC models requires the `esm` package "
            "(pip install esm)."
        )

    try:
        from esm.models.esmc import ESMC  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise EnvironmentError(
            "Loading tokenizer for EvolutionaryScale ESMC models requires the `esm` package "
            "(pip install esm)."
        ) from exc

    if "600m" in model_name:
        esmc_name = "esmc_600m"
    else:
        esmc_name = "esmc_300m"

    temp_model = ESMC.from_pretrained(esmc_name)
    tokenizer = temp_model.tokenizer
    del temp_model

    class _ESMCTokenizerShim:
        saprot_is_esmc = True

        def __init__(self, base_tokenizer):
            self._base = base_tokenizer

        def __call__(self, sequences, *args, **kwargs):
            if isinstance(sequences, str):
                sequences = [sequences]
            cleaned = [seq.replace(" ", "") for seq in sequences]
            proteins = [ESMProtein(sequence=seq) for seq in cleaned]
            return {"proteins": proteins}

        def __getattr__(self, item):
            return getattr(self._base, item)

    return _ESMCTokenizerShim(tokenizer)


def patch_auto_tokenizer_with_snapshot_fallback(cache_root: Path | None = None) -> None:
    """
    Wrap ``AutoTokenizer.from_pretrained`` so that, when a huggingface model ID is
    provided but there is a conflicting local directory missing tokenizer files,
    we fetch the tokenizer assets via ``snapshot_download`` and retry from the
    downloaded location.
    """

    if AutoTokenizer is None or snapshot_download is None:
        return

    original_from_pretrained = AutoTokenizer.from_pretrained
    if getattr(original_from_pretrained, "_saprot_patched", False):
        return

    cache_root = cache_root or _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    def _patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        try:
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except OSError as exc:
            if not _should_try_fallback(pretrained_model_name_or_path):
                raise

            repo_id = str(pretrained_model_name_or_path)

            if _is_esmc_repo(repo_id):
                return _load_esmc_tokenizer(repo_id)

            safe_dir = repo_id.replace("/", "__")
            local_dir = cache_root / safe_dir
            tokenizer_config = local_dir / "tokenizer_config.json"

            if not tokenizer_config.exists():
                local_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=[
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "vocab.txt",
                        "vocab.json",
                        "merges.txt",
                        "*.model",
                        "*.txt",
                        "config.json",
                    ],
                )

            return original_from_pretrained(str(local_dir), *args, **kwargs)

    _patched_from_pretrained._saprot_patched = True  # type: ignore[attr-defined]
    AutoTokenizer.from_pretrained = _patched_from_pretrained

