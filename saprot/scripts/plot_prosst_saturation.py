import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_from_json(input_json: str, output_png: str) -> str:
    with Path(input_json).open(encoding="utf-8") as handle:
        payload = json.load(handle)

    sequence = str(payload["sequence"])
    amino_acids = tuple(str(payload["amino_acids"]))
    values = np.asarray(payload["scores"], dtype=float)
    expected_shape = (len(amino_acids), len(sequence))
    if values.shape != expected_shape:
        raise ValueError(
            f"Saturation heatmap scores have shape {values.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Saturation heatmap scores contain non-finite values.")

    max_abs = float(np.abs(values).max())
    if max_abs == 0:
        max_abs = 1.0

    figure_width = min(40.0, max(8.0, len(sequence) * 0.18))
    figure, axis = plt.subplots(figsize=(figure_width, 7.0))
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axis.set_yticks(range(len(amino_acids)))
    axis.set_yticklabels(amino_acids)
    axis.set_ylabel("Mutant amino acid")

    tick_stride = max(1, math.ceil(len(sequence) / 50))
    tick_positions = list(range(0, len(sequence), tick_stride))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [f"{sequence[index]}{index + 1}" for index in tick_positions],
        rotation=90,
        fontsize=8,
    )
    axis.set_xlabel("Wild-type residue and position")
    axis.set_title("ProSST single-site saturation mutagenesis")
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label("log P(mutant) - log P(wild type)")
    figure.tight_layout()

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-png", required=True)
    args = parser.parse_args()
    plot_from_json(args.input_json, args.output_png)


if __name__ == "__main__":
    main()
