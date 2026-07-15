import json
from collections.abc import Sequence
from numbers import Integral
from typing import Any, List


RESIDUE_LABEL_IGNORE_INDEX = -100


def parse_residue_labels(value: Any) -> List[int]:
    """Parse residue labels from a JSON array or a whitespace/comma list."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Residue labels cannot be empty.")
        if text.startswith("["):
            try:
                values = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON residue_labels array.") from exc
        else:
            values = text.replace(",", " ").split()
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise ValueError(
            "residue_labels must be a JSON array or a whitespace-separated "
            "list of integer category IDs."
        )

    if not isinstance(values, list) or not values:
        raise ValueError("Residue labels must contain at least one category ID.")

    labels = []
    for label in values:
        if isinstance(label, bool):
            raise ValueError("Residue labels must be integer category IDs.")
        if isinstance(label, Integral):
            parsed = int(label)
        else:
            text = str(label).strip()
            try:
                parsed = int(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Residue labels must be integer category IDs; "
                    f"received {label!r}."
                ) from exc
            if text not in {str(parsed), f"+{parsed}"}:
                raise ValueError(
                    "Residue labels must be integer category IDs; "
                    f"received {label!r}."
                )

        if parsed < 0 and parsed != RESIDUE_LABEL_IGNORE_INDEX:
            raise ValueError(
                "Residue category IDs must be non-negative; use -100 only for "
                "residues that should be ignored."
            )
        labels.append(parsed)

    return labels


def validate_residue_labels(
    sequence: str,
    labels: Sequence[int],
    context: str = "",
) -> None:
    if len(sequence) != len(labels):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}residue_labels length must match sequence length: "
            f"labels={len(labels)}, sequence={len(sequence)}."
        )
    if all(label == RESIDUE_LABEL_IGNORE_INDEX for label in labels):
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}residue_labels cannot all be -100.")
