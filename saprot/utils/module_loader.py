import os
import copy
import pytorch_lightning as pl
import datetime
import wandb

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, Strategy

################################################################################
################################ load model ####################################
################################################################################
def my_load_model(config):
    model_config = copy.deepcopy(config)
    model_type = model_config.pop("model_py_path")

    if "kwargs" in model_config.keys():
        kwargs = model_config.pop('kwargs')
    else:
        kwargs = {}
    
    model_config.update(kwargs)

    # --- ESMC auto-redirect for saprot selections ---
    def _maybe_redirect_saprot_to_esmc(model_type_local: str, cfg: dict):
        cfg = copy.deepcopy(cfg)
        config_path = cfg.get("config_path", "")
        esmc_flag = isinstance(config_path, str) and ("esmc-" in config_path or "EvolutionaryScale/esmc-" in config_path)
        if not esmc_flag:
            return model_type_local, cfg

        # map config_path -> model_name
        model_name = "esmc_300m" if "esmc-300m" in config_path else ("esmc_600m" if "esmc-600m" in config_path else "esmc_300m")
        cfg.pop("config_path", None)
        cfg["model_name"] = model_name
        # remove unsupported keys for ESMC models
        cfg.pop("load_pretrained", None)
        # CRITICAL: DO NOT delete lora_kwargs! It is needed for LoRA to work.
        # keep lora_kwargs so LoRA can be applied by ESMCBaseModel

        # map saprot task to esmc task
        mapping = {
            "saprot/saprot_classification_model": "esmc/esmc_classification_model",
            "saprot/saprot_regression_model": "esmc/esmc_regression_model",
            "saprot/saprot_token_classification_model": "esmc/esmc_token_classification_model",
            "saprot/saprot_pair_classification_model": "esmc/esmc_pair_classification_model",
            "saprot/saprot_pair_regression_model": "esmc/esmc_pair_regression_model",
        }

        return mapping.get(model_type_local, model_type_local), cfg

    model_type, model_config = _maybe_redirect_saprot_to_esmc(model_type, model_config)

    if model_type == "saprot/saprot_classification_model":
      from model.saprot.saprot_classification_model import SaprotClassificationModel
      return SaprotClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_token_classification_model":
      from model.saprot.saprot_token_classification_model import SaprotTokenClassificationModel
      return SaprotTokenClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_regression_model":
      if 'num_labels' in model_config: del model_config['num_labels']
      from model.saprot.saprot_regression_model import SaprotRegressionModel
      return SaprotRegressionModel(**model_config)

    if model_type == "saprot/saprot_pair_classification_model":
      from model.saprot.saprot_pair_classification_model import SaprotPairClassificationModel
      return SaprotPairClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_pair_regression_model":
      if 'num_labels' in model_config: del model_config['num_labels']
      from model.saprot.saprot_pair_regression_model import SaprotPairRegressionModel
      return SaprotPairRegressionModel(**model_config)
    
    if model_type == "protT5/protT5_classification_model":
      from model.protT5.protT5_classification_model import ProtT5ClassificationModel
      return ProtT5ClassificationModel(**model_config)
    
    if model_type == "protT5/protT5_regression_model":
      from model.protT5.protT5_regression_model import ProtT5RegressionModel
      return ProtT5RegressionModel(**model_config)
    
    if model_type == "protT5/protT5_token_classification_model":
      from model.protT5.protT5_token_classification_model import ProtT5TokenClassificationModel
      return ProtT5TokenClassificationModel(**model_config)

    # ESMC models
    if model_type == "esmc/esmc_classification_model":
      from model.esmc.esmc_classification_model import ESMCClassificationModel
      return ESMCClassificationModel(**model_config)

    if model_type == "esmc/esmc_regression_model":
      from model.esmc.esmc_regression_model import ESMCRegressionModel
      return ESMCRegressionModel(**model_config)

    if model_type == "esmc/esmc_token_classification_model":
      from model.esmc.esmc_token_classification_model import ESMCTokenClassificationModel
      return ESMCTokenClassificationModel(**model_config)

    if model_type == "esmc/esmc_pair_classification_model":
      from model.esmc.esmc_pair_classification_model import ESMCPairClassificationModel
      return ESMCPairClassificationModel(**model_config)

    if model_type == "esmc/esmc_pair_regression_model":
      from model.esmc.esmc_pair_regression_model import ESMCPairRegressionModel
      return ESMCPairRegressionModel(**model_config)

    if model_type == "esmc/esmc_mutation_model":
      from model.esmc.esmc_mutation_model import ESMCMutationModel
      return ESMCMutationModel(**model_config)

    if model_type == "esmc/esmc_if_model":
      from model.esmc.esmc_if_model import ESMCIFModel
      return ESMCIFModel(**model_config)


################################################################################
################################ load dataset ##################################
################################################################################
def my_load_dataset(config):
    dataset_config = copy.deepcopy(config)
    dataset_type = dataset_config.pop("dataset_py_path")
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)

    # --- ESMC auto-redirect for saprot datasets ---
    def _maybe_redirect_dataset_to_esmc(dtype_local: str, cfg: dict):
        cfg = copy.deepcopy(cfg)
        tok = cfg.get("tokenizer", "")
        esmc_flag = isinstance(tok, str) and ("esmc-" in tok or "EvolutionaryScale/esmc-" in tok)
        if not esmc_flag:
            return dtype_local, cfg

        model_name = "esmc_300m" if "esmc-300m" in tok else ("esmc_600m" if "esmc-600m" in tok else "esmc_300m")
        cfg.pop("tokenizer", None)
        cfg["model_name"] = model_name
        # drop saprot-specific keys not accepted by LMDBDataset/__init__ chain
        for k in ["plddt_threshold", "mask_struc_ratio"]:
            if k in cfg:
                cfg.pop(k, None)

        mapping = {
            "saprot/saprot_classification_dataset": "esmc/esmc_classification_dataset",
            "saprot/saprot_token_classification_dataset": "esmc/esmc_token_classification_dataset",
            "saprot/saprot_regression_dataset": "esmc/esmc_regression_dataset",
            "saprot/saprot_pair_classification_dataset": "esmc/esmc_pair_classification_dataset",
            "saprot/saprot_pair_regression_dataset": "esmc/esmc_pair_regression_dataset",
            "saprot/saprot_annotation_dataset": "esmc/esmc_annotation_dataset",
        }

        return mapping.get(dtype_local, dtype_local), cfg

    dataset_type, dataset_config = _maybe_redirect_dataset_to_esmc(dataset_type, dataset_config)

    if dataset_type == "saprot/saprot_classification_dataset":
      from dataset.saprot.saprot_classification_dataset import SaprotClassificationDataset
      return SaprotClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_token_classification_dataset":
      if 'plddt_threshold' in dataset_config: del dataset_config['plddt_threshold']
      from dataset.saprot.saprot_token_classification_dataset import SaprotTokenClassificationDataset
      return SaprotTokenClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_regression_dataset":
      from dataset.saprot.saprot_regression_dataset import SaprotRegressionDataset
      return SaprotRegressionDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_pair_classification_dataset":
      from dataset.saprot.saprot_pair_classification_dataset import SaprotPairClassificationDataset
      return SaprotPairClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_pair_regression_dataset":
      from dataset.saprot.saprot_pair_regression_dataset import SaprotPairRegressionDataset
      return SaprotPairRegressionDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_classification_dataset":
      from dataset.protT5.protT5_classification_dataset import ProtT5ClassificationDataset
      return ProtT5ClassificationDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_regression_dataset":
      from dataset.protT5.protT5_regression_dataset import ProtT5RegressionDataset
      return ProtT5RegressionDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_token_classification_dataset":
      from dataset.protT5.protT5_token_classification_dataset import ProtT5TokenClassificationDataset
      return ProtT5TokenClassificationDataset(**dataset_config)

    # ESMC datasets
    if dataset_type == "esmc/esmc_classification_dataset":
      from dataset.esmc.esmc_classification_dataset import ESMCClassificationDataset
      return ESMCClassificationDataset(**dataset_config)

    if dataset_type == "esmc/esmc_annotation_dataset":
      from dataset.esmc.esmc_annotation_dataset import ESMCAnnotationDataset
      return ESMCAnnotationDataset(**dataset_config)

    if dataset_type == "esmc/esmc_token_classification_dataset":
      from dataset.esmc.esmc_token_classification_dataset import ESMCTokenClassificationDataset
      return ESMCTokenClassificationDataset(**dataset_config)

    if dataset_type == "esmc/esmc_pair_classification_dataset":
      from dataset.esmc.esmc_pair_classification_dataset import ESMCPairClassificationDataset
      return ESMCPairClassificationDataset(**dataset_config)

    if dataset_type == "esmc/esmc_pair_regression_dataset":
      from dataset.esmc.esmc_pair_regression_dataset import ESMCPairRegressionDataset
      return ESMCPairRegressionDataset(**dataset_config)

    if dataset_type == "esmc/esmc_regression_dataset":
      from dataset.esmc.esmc_regression_dataset import ESMCRegressionDataset
      return ESMCRegressionDataset(**dataset_config)

def load_wandb(config):
    # initialize wandb
    wandb_config = config.setting.wandb_config
    wandb_logger = WandbLogger(project=wandb_config.project, config=config,
                               name=wandb_config.name,
                               settings=wandb.Settings())
    
    return wandb_logger


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)
    
    if "kwargs" in model_config.keys():
        kwargs = model_config.pop('kwargs')
    else:
        kwargs = {}
        
    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)
    
    if "kwargs" in dataset_config.keys():
        kwargs = dataset_config.pop('kwargs')
    else:
        kwargs = {}
        
    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# def load_plugins():
#     config = get_config()
#     # initialize plugins
#     plugins = []
#
#     if "Trainer_plugin" not in config.keys():
#         return plugins
#
#     if not config.Trainer.logger:
#         if hasattr(config.Trainer_plugin, "LearningRateMonitor"):
#             config.Trainer_plugin.pop("LearningRateMonitor", None)
#
#     if not config.Trainer.enable_checkpointing:
#         if hasattr(config.Trainer_plugin, "ModelCheckpoint"):
#             config.Trainer_plugin.pop("ModelCheckpoint", None)
#
#     for plugin, kwargs in config.Trainer_plugin.items():
#         plugins.append(eval(plugin)(**kwargs))
#
#     return plugins


# Initialize strategy
def load_strategy(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop('timeout'))
        config["timeout"] = datetime.timedelta(seconds=timeout)
    
    cls = config.pop('class')
    return eval(cls)(**config)


# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Fix for ESMC models: force FP16 instead of BFloat16 for mixed precision
    # BFloat16 has compatibility issues with certain CUDA operations
    if hasattr(config, 'model') and hasattr(config.model, 'model_py_path'):
        model_path = config.model.model_py_path
        if 'esmc' in model_path.lower() and trainer_config.get('precision') == '16-mixed':
            # Force FP16 instead of auto-selecting BFloat16
            trainer_config['precision'] = '16'
            print(f"ESMC model detected: Changed precision from '16-mixed' to '16' (FP16) to avoid BFloat16 compatibility issues")
    
    # Initialize wandb
    if trainer_config.logger:
        trainer_config.logger = load_wandb(config)
    else:
        trainer_config.logger = False
    
    # Initialize strategy
    # strategy = load_strategy(trainer_config.pop('strategy'))
    # Strategy is not used in Colab
    trainer_config.pop('strategy')
    
    return pl.Trainer(**trainer_config, callbacks=[], use_distributed_sampler=False)