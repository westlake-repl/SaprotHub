#!/usr/bin/env python
# coding: utf-8

# ### Load Dataset with T5Tokenizer

# In[ ]:


# Download https://github.com/google/sentencepiece#installation


import sys
sys.path.append('/sujin/PycharmProjects/zhangxuting/workspace/ColabProtT5/saprot')

dataset_config = {
    "dataloader_kwargs": {
        "batch_size": 1,
        "num_workers": 2
    },
    "train_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/train",
    "valid_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/valid",
    "test_lmdb": "/sujin/Datasets/LMDB/DeepLoc/cls2/normal_splits/foldseek/test",
    "tokenizer": "/sujin/Models/ProtTrans/prot_t5_xl_uniref50",
    "plddt_threshold": None
}

from dataset.protT5.protT5_classification_dataset import ProtT5ClassificationDataset
data_module = ProtT5ClassificationDataset(**dataset_config)


# ### Load T5 Model
# 

# In[2]:


model_config = {
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

print(model_config)

from model.protT5.protT5_classification_model import ProtT5ClassificationModel
model_module = ProtT5ClassificationModel(**model_config["kwargs"])


# In[3]:


import pytorch_lightning as pl

trainer_config = {
    "max_epochs": 5,
    "log_every_n_steps": 1,
    # "strategy":{
    #     "class": "auto"
    # },
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

trainer = pl.Trainer(**trainer_config, callbacks=[], use_distributed_sampler=False)


# In[5]:


trainer.fit(model=model_module, datamodule=data_module)

