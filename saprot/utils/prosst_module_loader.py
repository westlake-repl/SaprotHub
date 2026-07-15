import copy

from saprot.utils.module_loader import load_trainer


def my_load_model(config):
    model_config = copy.deepcopy(config)
    model_type = model_config.pop("model_py_path")
    model_config.update(model_config.pop("kwargs", {}))

    if model_type == "prosst/prosst_classification_model":
        from saprot.model.prosst.prosst_classification_model import (
            ProSSTClassificationModel,
        )

        return ProSSTClassificationModel(**model_config)

    if model_type == "prosst/prosst_regression_model":
        model_config.pop("num_labels", None)
        from saprot.model.prosst.prosst_regression_model import ProSSTRegressionModel

        return ProSSTRegressionModel(**model_config)

    if model_type == "prosst/prosst_token_classification_model":
        from saprot.model.prosst.prosst_token_classification_model import (
            ProSSTTokenClassificationModel,
        )

        return ProSSTTokenClassificationModel(**model_config)

    if model_type == "prosst/prosst_pair_classification_model":
        from saprot.model.prosst.prosst_pair_classification_model import (
            ProSSTPairClassificationModel,
        )

        return ProSSTPairClassificationModel(**model_config)

    if model_type == "prosst/prosst_pair_regression_model":
        model_config.pop("num_labels", None)
        from saprot.model.prosst.prosst_pair_regression_model import (
            ProSSTPairRegressionModel,
        )

        return ProSSTPairRegressionModel(**model_config)

    if model_type == "prosst/prosst_mutation_model":
        from saprot.model.prosst.prosst_mutation_model import ProSSTMutationModel

        return ProSSTMutationModel(**model_config)

    raise ValueError(f"Unsupported ProSST model type: {model_type}")


def my_load_dataset(config):
    dataset_config = copy.deepcopy(config)
    dataset_type = dataset_config.pop("dataset_py_path")
    dataset_config.update(dataset_config.pop("kwargs", {}))

    if dataset_type == "prosst/prosst_classification_dataset":
        from saprot.dataset.prosst.prosst_classification_dataset import (
            ProSSTClassificationDataset,
        )

        return ProSSTClassificationDataset(**dataset_config)

    if dataset_type == "prosst/prosst_regression_dataset":
        from saprot.dataset.prosst.prosst_regression_dataset import (
            ProSSTRegressionDataset,
        )

        return ProSSTRegressionDataset(**dataset_config)

    if dataset_type == "prosst/prosst_token_classification_dataset":
        from saprot.dataset.prosst.prosst_token_classification_dataset import (
            ProSSTTokenClassificationDataset,
        )

        return ProSSTTokenClassificationDataset(**dataset_config)

    if dataset_type == "prosst/prosst_pair_classification_dataset":
        from saprot.dataset.prosst.prosst_pair_classification_dataset import (
            ProSSTPairClassificationDataset,
        )

        return ProSSTPairClassificationDataset(**dataset_config)

    if dataset_type == "prosst/prosst_pair_regression_dataset":
        from saprot.dataset.prosst.prosst_pair_regression_dataset import (
            ProSSTPairRegressionDataset,
        )

        return ProSSTPairRegressionDataset(**dataset_config)

    if dataset_type == "prosst/prosst_mutation_dataset":
        from saprot.dataset.prosst.prosst_mutation_dataset import ProSSTMutationDataset

        return ProSSTMutationDataset(**dataset_config)

    raise ValueError(f"Unsupported ProSST dataset type: {dataset_type}")
