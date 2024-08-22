import pytorch_lightning as pl
from easydict import EasyDict
import sys
sys.path.append('/sujin/PycharmProjects/zhangxuting/workspace/ColabProtT5')
from saprot.scripts.training import finetune



setting_config = {
    "seed": 42,
    "os_environ": {
        "CUDA_VISIBLE_DEVICES": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "8888",
        "NODE_RANK": 0,
        "WORLD_SIZE": 1
    }
}

dataset_config = {
    "dataset_py_path": "protT5/protT5_classification_dataset",
    "dataloader_kwargs": {
        "batch_size": 1,
        "num_workers": 2
    },
    "train_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/train",
    "valid_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/valid",
    "test_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/test",
    "tokenizer": "/sujin/Models/ProtTrans/prot_t5_xl_uniref50",
    "kwargs": {
        "plddt_threshold": 0.5,
    }
}

model_config = {
    "model_py_path": "protT5/protT5_classification_model",
    "kwargs":{
        "config_path": "/sujin/Models/ProtTrans/prot_t5_xl_uniref50", 
        "load_pretrained": True,
        "num_labels": 2,
        'lora_kwargs': {
            'config_list': [],
            'is_trainable': True,
            'lora_alpha': 16,
            'lora_dropout': 0.0,
            'num_lora': 1,
            'r': 8
    }
    },
    "lr_scheduler_kwargs": {
        "class": "ConstantLRScheduler",
        "init_lr": 1e-4
    },

}

trainer_config = {
    "max_epochs": 5,
    "log_every_n_steps": 1,
    "strategy":{
        "class": "auto"
    },
    "logger": False,
    "enable_checkpointing": False,
    "val_check_interval": 0.25,
    "accelerator": "gpu",
    "devices": 1,
    "num_nodes": 1,
    "accumulate_grad_batches": 1,
    "precision": 16,
    "num_sanity_val_steps": 0
}

config_dict = {
    "setting": setting_config,
    "model": model_config,
    "dataset": dataset_config,
    "Trainer": trainer_config
}

config = EasyDict(config_dict)

print(config.setting)



finetune(config)
