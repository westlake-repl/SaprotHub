import torch
import os

from typing import List, Dict
from data.pdb2feature import batch_coords2feature
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    EsmForMaskedLM,
    EsmForSequenceClassification,
)
from easydict import EasyDict
from ..abstract_model import AbstractModel

import matplotlib.pyplot as plt


class SaprotBaseModel(AbstractModel):
    """
    ESM base model. It cannot be used directly but provides model initialization for downstream tasks.
    """
    
    def __init__(self,
                 task: str,
                 config_path: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 gradient_checkpointing: bool = False,
                 lora_kwargs: dict = None,
                 **kwargs):
        """
        Args:
            task: Task name。

            config_path: Path to the config file of huggingface esm model

            extra_config: Extra config for the model

            load_pretrained: Whether to load pretrained weights of base model

            freeze_backbone: Whether to freeze the backbone of the model

            gradient_checkpointing: Whether to enable gradient checkpointing

            lora_kwargs: LoRA configuration

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ['classification', 'token_classification', 'regression', 'lm', 'base']
        self.task = task
        self.config_path = config_path
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs
        super().__init__(**kwargs)
        
        # After all initialization done, lora technique is applied if needed
        if self.lora_kwargs is not None:
            # No need to freeze backbone if LoRA is used
            self.freeze_backbone = False
            
            self.lora_kwargs = EasyDict(lora_kwargs)
            self._init_lora()
        
        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []
    
    def _init_lora(self):
        from peft import (
            LoraConfig,
            # PeftModelForSequenceClassification,
            # get_peft_model
        )
        
        from .self_peft.mapping import get_peft_model
        from .self_peft.peft_model import PeftModelForSequenceClassification
        
        is_trainable = getattr(self.lora_kwargs, "is_trainable", False)
        config_list = getattr(self.lora_kwargs, "config_list", [])
        assert self.lora_kwargs.num_lora >= len(config_list), ("The number of LoRA models should be greater than or "
                                                               "equal to the number of weight files.")
        for i in range(self.lora_kwargs.num_lora):
            adapter_name = f"adapter_{i}" if self.lora_kwargs.num_lora > 1 else "default"
            
            # Load pre-trained LoRA weights
            if i < len(config_list):
                lora_config_path = config_list[i].lora_config_path
                if i == 0:
                    # If i == 0, initialize a PEFT model
                    self.model = PeftModelForSequenceClassification.from_pretrained(self.model,
                                                                                    lora_config_path,
                                                                                    adapter_name=adapter_name,
                                                                                    is_trainable=is_trainable)
                else:
                    self.model.load_adapter(lora_config_path, adapter_name=adapter_name, is_trainable=is_trainable)
            
            # Initialize LoRA model for training
            else:
                lora_config = {
                    "task_type": "SEQ_CLS",
                    "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
                    "modules_to_save": ["classifier"],
                    "inference_mode": False,
                    "r": getattr(self.lora_kwargs, "r", 8),
                    "lora_dropout": getattr(self.lora_kwargs, "lora_dropout", 0.0),
                    "lora_alpha": getattr(self.lora_kwargs, "lora_alpha", 16),
                }
                
                lora_config = LoraConfig(**lora_config)
                
                if i == 0:
                    # If i == 0, initialize a PEFT model
                    self.model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
                
                else:
                    self.model.add_adapter(adapter_name, lora_config)
        
        if self.lora_kwargs.num_lora > 1:
            # Multiple LoRA models only support inference mode
            print("Multiple LoRA models are used. This only supports inference mode. If you want to train the model,"
                  "set num_lora to 1.")
            
            # Replace the normal forward function with the lora ensemble function, which averages the outputs of all
            # LoRA models.
            def lora_forward(func):
                
                def forward(*args, **kwargs):
                    logits_list = []
                    ori_shape = None
                    
                    for i in range(self.lora_kwargs.num_lora):
                        adapter_name = f"adapter_{i}"
                        self.model.set_adapter(adapter_name)
                        logits = func(*args, **kwargs)
                        logits_list.append(logits)
                        
                        if ori_shape is None:
                            ori_shape = logits.shape
                    
                    logits = torch.stack(logits_list, dim=0)
                    
                    # For classification task, final labels are voted by all LoRA models
                    if len(ori_shape) == 2:
                        logits = logits.permute(1, 0, 2)
                        preds = logits.argmax(dim=-1)
                        preds = torch.mode(preds, dim=1).values
                        
                        # Generate dummy logits to match the original output
                        dummy_logits = torch.zeros(ori_shape).to(logits)
                        for i, pred in enumerate(preds):
                            dummy_logits[i, pred] = 1.0
                    
                    # For regression task, final labels are averaged among all LoRA models
                    else:
                        dummy_logits = logits.mean(dim=0)
                    
                    return dummy_logits.detach()
                
                return forward
            
            self.forward = lora_forward(self.forward)
        
        print(f"Now active LoRA model: {self.model.active_adapter}")
        self.model.print_trainable_parameters()
        
        # After LoRA model is initialized, add trainable parameters to optimizer)
        self.init_optimizers()
    
    def initialize_model(self):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_path)
        
        # Initialize different models according to task
        config = AutoConfig.from_pretrained(self.config_path)
        if self.extra_config:
            for k, v in self.extra_config.items():
                setattr(config, k, v)
        
        else:
            self.extra_config = {}
        
        if self.task == 'classification':
            # Note that self.num_labels should be set in child classes
            if self.load_pretrained:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=self.num_labels, **self.extra_config)
            
            else:
                config.num_labels = self.num_labels
                self.model = AutoModelForSequenceClassification.from_config(config)
        
        if self.task == 'token_classification':
            # Note that self.num_labels should be set in child classes
            if self.load_pretrained:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.config_path, num_labels=self.num_labels, **self.extra_config)
            
            else:
                config.num_labels = self.num_labels
                self.model = AutoModelForTokenClassification.from_config(config)
        
        elif self.task == 'regression':
            if self.load_pretrained:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=1, **self.extra_config)
            
            else:
                config.num_labels = 1
                self.model = AutoModelForSequenceClassification.from_config(config)
        
        elif self.task == 'lm':
            if self.load_pretrained:
                self.model = AutoModelForMaskedLM.from_pretrained(self.config_path, **self.extra_config)
            
            else:
                self.model = AutoModelForMaskedLM.from_config(config)
        
        elif self.task == 'base':
            if self.load_pretrained:
                self.model = AutoModelForMaskedLM.from_pretrained(self.config_path, **self.extra_config)
            
            else:
                self.model = AutoModelForMaskedLM.from_config(config)
            
            if isinstance(self.model, EsmForMaskedLM) or isinstance(self.model, EsmForSequenceClassification):
                self.model.lm_head = None
        
        if isinstance(self.model, EsmForMaskedLM) or isinstance(self.model, EsmForSequenceClassification):
            # Remove contact head
            self.model.esm.contact_head = None
            
            # Remove position embedding if the embedding type is ``rotary``
            if config.position_embedding_type == "rotary":
                self.model.esm.embeddings.position_embeddings = None
            
            # Set gradient checkpointing
            self.model.esm.encoder.gradient_checkpointing = self.gradient_checkpointing
        
        # Freeze the backbone of the model
        if self.freeze_backbone:
            for param in self.model.esm.parameters():
                param.requires_grad = False
        
        # # Disable the pooling layer
        # backbone = getattr(self.model, "esm", self.model.bert)
        # backbone.pooler = None
    
    def initialize_metrics(self, stage: str) -> dict:
        return {}
    
    def get_hidden_states_from_dict(self, inputs: dict, reduction: str = None) -> list:
        """
        Get hidden representations from input dict

        Args:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        inputs["output_hidden_states"] = True
        outputs = self.model.esm(**inputs)
        
        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)
        
        repr_list = []
        hidden_states = outputs["hidden_states"][-1]
        for i, idx in enumerate(indices):
            if reduction == "mean":
                repr = hidden_states[i][1:idx].mean(dim=0)
            else:
                repr = hidden_states[i][1:idx]
            
            repr_list.append(repr)
        
        return repr_list
    
    def get_hidden_states_from_seqs(self, seqs: list, reduction: str = None) -> list:
        """
        Get hidden representations of protein sequences

        Args:
            seqs: A list of protein sequences
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        inputs = self.tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs["output_hidden_states"] = True
        outputs = self.model.esm(**inputs)
        
        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)
        
        repr_list = []
        hidden_states = outputs["hidden_states"][-1]
        for i, idx in enumerate(indices):
            if reduction == "mean":
                repr = hidden_states[i][1:idx].mean(dim=0)
            else:
                repr = hidden_states[i][1:idx]
            
            repr_list.append(repr)
        
        return repr_list
    
    def add_bias_feature(self, inputs, coords: List[Dict]) -> torch.Tensor:
        """
        Add structure information as biases to attention map. This function is used to add structure information
        to the model as Evoformer does.

        Args:
            inputs: A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            coords: Coordinates of backbone atoms. Each element is a dictionary with keys ["N", "CA", "C", "O"].

        Returns
            pair_feature: A tensor of shape [B, L, L, 407]. Here 407 is the RBF of distance(400) + angle(7).
        """
        inputs["pair_feature"] = batch_coords2feature(coords, self.model.device)
        return inputs
    
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        Rewrite this function to save LoRA parameters
        """
        
        if not self.lora_kwargs:
            return super().save_checkpoint(save_path, save_info, save_weights_only)
        
        else:
            try:
                if hasattr(self.trainer.strategy, "deepspeed_engine"):
                    save_path = os.path.dirname(save_path)
            except Exception as e:
                pass
            
            self.model.save_pretrained(save_path)
    
    def output_test_metrics(self, log_dict):
        # Remove valid_loss from log_dict when the task is classification
        if "test_acc" in log_dict:
            log_dict.pop("test_loss")
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and self.num_labels > 2:
            log_dict.pop("test_mcc")
        
        METRIC_MAP = {
            "test_acc": "Classification accuracy (Acc)",
            "test_loss": "Root mean squared error (RMSE)",  # Only for regression task
            "test_mcc": "Matthews correlation coefficient (MCC)",
            "test_r2": "Coefficient of determination (R^2)",
            "test_spearman": "Spearman correlation",
            "test_pearson": "Pearson correlation",
        }
        
        print('=' * 100)
        print('Evaluation results on the test set:')
        flag = False
        for key, value in log_dict.items():
            if value is not None:
                print_value = value.item()
            else:
                print_value = torch.nan
                flag = True
            
            print(f"{METRIC_MAP[key.lower()]}: {print_value}")
        
        if "classification" not in self.task and flag:
            print("\033[31m\nWarning: To calculate some metrics (R^2, Spearman correlation, Pearson correlation), "
                  "a minimum of two examples from the validation/test set is required.\033[0m")
        print('=' * 100)
    
    def plot_valid_metrics_curve(self, log_dict):
        if not hasattr(self, 'grid'):
            from google.colab import widgets
            width = 400 * len(log_dict)
            height = 400
            self.grid = widgets.Grid(1, 1, header_row=False, header_column=False,
                                     style=f'width:{width}px; height:{height}px')
        
        # Remove valid_loss from log_dict when the task is classification
        if "valid_acc" in log_dict:
            log_dict.pop("valid_loss")
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and self.num_labels > 2:
            log_dict.pop("valid_mcc")
        
        METRIC_MAP = {
            "valid_acc": "Classification accuracy (Acc)",
            "valid_loss": "Root mean squared error (RMSE)",  # Only for regression task
            "valid_mcc": "Matthews correlation coefficient (MCC)",
            "valid_r2": "Coefficient of determination (R$^2$)",
            "valid_spearman": "Spearman correlation",
            "valid_pearson": "Pearson correlation",
        }
        
        with self.grid.output_to(0, 0):
            self.grid.clear_cell()
            
            fig = plt.figure(figsize=(6 * len(log_dict), 6))
            ax = []
            self.valid_metrics_list['step'].append(int(self.step))
            for idx, metric in enumerate(log_dict.keys()):
                value = torch.nan if log_dict[metric] is None else log_dict[metric].detach().cpu().item()
                
                if metric in self.valid_metrics_list:
                    self.valid_metrics_list[metric].append(value)
                else:
                    self.valid_metrics_list[metric] = [value]
                
                ax.append(fig.add_subplot(1, len(log_dict), idx + 1))
                ax[idx].set_title(METRIC_MAP[metric.lower()])
                ax[idx].set_xlabel('step')
                ax[idx].set_ylabel(METRIC_MAP[metric.lower()])
                ax[idx].plot(self.valid_metrics_list['step'], self.valid_metrics_list[metric], marker='o')
            
            import ipywidgets
            import markdown
            from IPython.display import display
            
            hint = ipywidgets.HTML(
                markdown.markdown(
                    f"### The model is saved to {self.save_path}.\n\n"
                    "### Evaluation results on the validation set are shown below.\n\n"
                    "### You can check <a href='https://github.com/westlake-repl/SaprotHub/wiki/SaprotHub-v2-(latest)#3-how-can-i-monitor-model-performance-during-training-and-detect-overfitting' target='blank'>here</a> to see how to judge the overfitting of your model."
                )
            )
            display(hint)
            plt.tight_layout()
            plt.show()
