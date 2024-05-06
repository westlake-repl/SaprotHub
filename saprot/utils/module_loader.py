import sys
import os
current_file = os.path.abspath(__file__)
saprot_dir = os.path.dirname(current_file)
colabsaprot_dir = os.path.dirname(saprot_dir)
sys.path.append(colabsaprot_dir)

import os
import copy
import pytorch_lightning as pl
import datetime
import wandb

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, DataParallelStrategy


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

    if model_type == "esm/esm_classification_model":
      from model.esm.esm_classification_model import EsmClassificationModel
      return EsmClassificationModel(**model_config)
    
    if model_type == "esm/esm_token_classification_model":
      from model.esm.esm_token_classification_model import EsmTokenClassificationModel
      return EsmTokenClassificationModel(**model_config)
    
    if model_type == "esm/esm_regression_model":
      if 'num_labels' in model_config: del model_config['num_labels']
      from model.esm.esm_regression_model import EsmRegressionModel
      return EsmRegressionModel(**model_config)


################################################################################
################################ load dataset ##################################
################################################################################
def my_load_dataset(config):
    dataset_config = copy.deepcopy(config)
    dataset_type = dataset_config.pop("dataset_py_path")
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)

    if dataset_type == "esm/esm_classification_dataset":
      from dataset.esm.esm_classification_dataset import EsmClassificationDataset
      return EsmClassificationDataset(**dataset_config)
    if dataset_type == "esm/esm_token_classification_dataset":
      if 'plddt_threshold' in dataset_config: del dataset_config['plddt_threshold']
      from dataset.esm.esm_token_classification_dataset import EsmTokenClassificationDataset
      return EsmTokenClassificationDataset(**dataset_config)
    if dataset_type == "esm/esm_regression_dataset":
      from dataset.esm.esm_regression_dataset import EsmRegressionDataset
      return EsmRegressionDataset(**dataset_config)
    if dataset_type == "esm/esm_pair_regression_dataset":
      from dataset.esm.esm_pair_regression_dataset import EsmPairRegressionDataset
      return EsmPairRegressionDataset(**dataset_config)

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
    
    if 'class' in config:
        cls = config.pop('class')
        return eval(cls)(**config)
    else:
        return None
        


# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize wandb
    if trainer_config.logger:
        trainer_config.logger = load_wandb(config)
    else:
        trainer_config.logger = False
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=[])
