from dataclasses import dataclass
from typing import Optional


PROSST_HUB_NAMESPACE = "ProSSTHub"
PROSST_HUB_URL = f"https://huggingface.co/{PROSST_HUB_NAMESPACE}"


@dataclass(frozen=True)
class ProSSTModelSpec:
    model_path: str
    structure_vocab_size: int

    @property
    def display_name(self) -> str:
        return f"Official ProSST ({self.structure_vocab_size})"

    @property
    def encoded_structure_vocab_size(self) -> int:
        return self.structure_vocab_size + 3


PROSST_MODEL_SPECS = tuple(
    ProSSTModelSpec(
        model_path=f"AI4Protein/ProSST-{structure_vocab_size}",
        structure_vocab_size=structure_vocab_size,
    )
    for structure_vocab_size in (20, 128, 512, 1024, 2048, 4096)
)
PROSST_MODELS_BY_PATH = {spec.model_path: spec for spec in PROSST_MODEL_SPECS}
PROSST_MODELS_BY_VOCAB_SIZE = {
    spec.structure_vocab_size: spec for spec in PROSST_MODEL_SPECS
}

DEFAULT_PROSST_MODEL = PROSST_MODELS_BY_VOCAB_SIZE[2048]
MODEL_PROSST_20 = PROSST_MODELS_BY_VOCAB_SIZE[20].model_path
MODEL_PROSST_128 = PROSST_MODELS_BY_VOCAB_SIZE[128].model_path
MODEL_PROSST_512 = PROSST_MODELS_BY_VOCAB_SIZE[512].model_path
MODEL_PROSST_1024 = PROSST_MODELS_BY_VOCAB_SIZE[1024].model_path
MODEL_PROSST_2048 = DEFAULT_PROSST_MODEL.model_path
MODEL_PROSST_4096 = PROSST_MODELS_BY_VOCAB_SIZE[4096].model_path


def get_prosst_model_spec(model_path: str) -> ProSSTModelSpec:
    normalized_path = str(model_path).strip()
    try:
        return PROSST_MODELS_BY_PATH[normalized_path]
    except KeyError as exc:
        expected = ", ".join(PROSST_MODELS_BY_PATH)
        raise ValueError(
            f"Unsupported official ProSST model: {normalized_path!r}. "
            f"Expected one of: {expected}."
        ) from exc


def resolve_structure_vocab_size(
    model_path: str,
    structure_vocab_size: Optional[int] = None,
) -> int:
    spec = PROSST_MODELS_BY_PATH.get(str(model_path).strip())
    if spec is None:
        if structure_vocab_size is None:
            raise ValueError(
                "A custom ProSST model requires an explicit structure_vocab_size."
            )
        return int(structure_vocab_size)

    if (
        structure_vocab_size is not None
        and int(structure_vocab_size) != spec.structure_vocab_size
    ):
        raise ValueError(
            f"{spec.model_path} requires structure_vocab_size="
            f"{spec.structure_vocab_size}, got {structure_vocab_size}."
        )
    return spec.structure_vocab_size
