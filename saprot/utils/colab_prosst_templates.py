INPUT_TEMPLATE_FILES = {
    "training": {
        "classification": {
            "sequence": "prosst_classification_sequence_template.csv",
            "structure": "prosst_classification_structure_template.csv",
        },
        "regression": {
            "sequence": "prosst_regression_sequence_template.csv",
            "structure": "prosst_regression_structure_template.csv",
        },
        "token_classification": {
            "sequence": "prosst_token_classification_sequence_template.csv",
            "structure": "prosst_token_classification_structure_template.csv",
        },
        "pair_classification": {
            "sequence": "prosst_pair_classification_sequence_template.csv",
            "structure": "prosst_pair_classification_structure_template.csv",
        },
        "pair_regression": {
            "sequence": "prosst_pair_regression_sequence_template.csv",
            "structure": "prosst_pair_regression_structure_template.csv",
        },
    },
    "prediction": {
        "single": {
            "sequence": "prosst_prediction_sequence_template.csv",
            "structure": "prosst_prediction_structure_template.csv",
        },
        "pair": {
            "sequence": "prosst_pair_prediction_sequence_template.csv",
            "structure": "prosst_pair_prediction_structure_template.csv",
        },
    },
    "embedding": {
        "single": {
            "sequence": "prosst_embedding_sequence_template.csv",
            "structure": "prosst_embedding_structure_template.csv",
        },
    },
    "zero_shot": {
        "single": {
            "sequence": "prosst_zero_shot_sequence_template.csv",
            "structure": "prosst_zero_shot_structure_template.csv",
        },
    },
    "saturation": {
        "single": {
            "sequence": "prosst_saturation_sequence_template.csv",
            "structure": "prosst_saturation_structure_template.csv",
        },
    },
}

INPUT_TEMPLATE_GUIDE = (
    (
        "TRAINING",
        (
            ("training", "classification", "Protein-level Classification"),
            ("training", "regression", "Protein-level Regression"),
            (
                "training",
                "token_classification",
                "Residue-level Classification",
            ),
            (
                "training",
                "pair_classification",
                "Protein-pair Classification",
            ),
            ("training", "pair_regression", "Protein-pair Regression"),
        ),
    ),
    (
        "PREDICTION AND ANALYSIS",
        (
            ("prediction", "single", "Single-protein property prediction"),
            ("prediction", "pair", "Protein-pair property prediction"),
            ("embedding", "single", "Embedding extraction"),
            ("zero_shot", "single", "Mutational effect prediction"),
            (
                "saturation",
                "single",
                "Single-site saturation mutagenesis",
            ),
        ),
    ),
)


def get_input_template_name(group: str, task: str, input_mode: str) -> str:
    try:
        return INPUT_TEMPLATE_FILES[group][task][input_mode]
    except KeyError as exc:
        raise ValueError(
            "Unknown ColabProSST input-template selection: "
            f"group={group}, task={task}, input_mode={input_mode}."
        ) from exc
