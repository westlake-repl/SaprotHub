import torch
import os

from easydict import EasyDict
from ..abstract_model import AbstractModel

import matplotlib.pyplot as plt

try:
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


class ESMCBaseModel(AbstractModel):
    """
    ESMC base model. Provides model initialization for downstream tasks.
    """
    
    def __init__(self,
                 task: str,
                 model_name: str = "esmc_300m",
                 device: str = None,
                 freeze_backbone: bool = False,
                 gradient_checkpointing: bool = False,
                 lora_kwargs: dict = None,
                 **kwargs):
        """
        Args:
            task: Task name.

            model_name: Model name for ESMC (esmc_300m or esmc_600m)

            device: Device to use (cuda or cpu)

            freeze_backbone: Whether to freeze the backbone of the model

            gradient_checkpointing: Whether to enable gradient checkpointing

            lora_kwargs: LoRA configuration

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ["classification", "token_classification", "regression"]
        self.task = task
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs

        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        super().__init__(**kwargs)
        
        # After all initialization done, lora technique is applied if needed
        if self.lora_kwargs is not None:
            # No need to freeze backbone if LoRA is used
            self.freeze_backbone = False
            
            self.lora_kwargs = EasyDict(self.lora_kwargs)
            self._init_lora()
        
        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []
    
    def _init_lora(self):
        from peft import LoraConfig, get_peft_model
        
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
                    self.model = get_peft_model(self.model, lora_config_path, adapter_name=adapter_name)
                else:
                    self.model.load_adapter(lora_config_path, adapter_name=adapter_name, is_trainable=is_trainable)
            
            # Initialize LoRA model for training
            else:
                # ESMC-specific default targets
                target_modules = getattr(
                    self.lora_kwargs,
                    "target_modules",
                    [
                        "layernorm_qkv.1", 
                        "out_proj",
                        "ffn.1",
                        "ffn.3",
                    ],
                )
                lora_config = {
                    "task_type": "FEATURE_EXTRACTION",
                    "target_modules": target_modules,
                    "modules_to_save": ["classifier"],
                    "inference_mode": False,
                    "r": getattr(self.lora_kwargs, "r", 8),
                    "lora_dropout": getattr(self.lora_kwargs, "lora_dropout", 0.0),
                    "lora_alpha": getattr(self.lora_kwargs, "lora_alpha", 16),
                    "bias": "none",
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
        
        # Custom parameter counting with deduplication to avoid PEFT modules_to_save double counting
        unique_params = set()
        total_params = 0
        trainable_params = 0
        lora_params = 0
        classifier_params = 0
        
        for n, p in self.model.named_parameters():
            param_id = id(p)
            if param_id not in unique_params:
                unique_params.add(param_id)
                num = p.numel()
                total_params += num
                if p.requires_grad:
                    trainable_params += num
                    if "lora_" in n:
                        lora_params += num
                    elif "classifier" in n:
                        classifier_params += num
        
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100.0 * trainable_params / total_params if total_params > 0 else 0:.6f}")
        print(f"  LoRA params: {lora_params:,}, Classifier params: {classifier_params:,}")
        
        # Ensure classifier is trainable
        if hasattr(self.model, 'classifier'):
            for n, p in self.model.classifier.named_parameters():
                p.requires_grad = True
        # Freeze non-LoRA, non-classifier params (align with Saprot behavior: only update LoRA+head)
        for n, p in self.model.named_parameters():
            is_lora = ("lora_" in n)
            # PEFT may prefix module paths; be robust by checking substring
            is_classifier = ("classifier" in n)
            p.requires_grad = is_lora or is_classifier
        
        # After LoRA model is initialized, add trainable parameters to optimizer)
        self.init_optimizers()
    
    def initialize_model(self):
        try:
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            for _prop in ["cls_token", "pad_token", "mask_token", "eos_token"]:
                if hasattr(EsmSequenceTokenizer, _prop):
                    try:
                        delattr(EsmSequenceTokenizer, _prop)
                    except Exception:
                        pass
        except Exception:
            pass

        # Load ESMC backbone and tokenizer
        # Note: Don't move model to device here - let PyTorch Lightning manage device placement
        # to avoid conflicts with mixed precision training (16-mixed)
        self.model = ESMC.from_pretrained(self.model_name)
        self.model = self.model.to(torch.float32)
        self.tokenizer = self.model.tokenizer

        # Attach simple heads per task
        # Get hidden_size: try multiple methods to be robust
        hidden_size = None
        # Method 1: Try to get from config
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        # Method 2: Try to get from embedding layer dimension
        elif hasattr(self.model, 'embed') and hasattr(self.model.embed, 'embedding_dim'):
            hidden_size = self.model.embed.embedding_dim
        # Method 3: Infer from model_name (fallback)
        if hidden_size is None:
            if '600m' in self.model_name.lower() or '600' in self.model_name:
                hidden_size = 1152  # ESMC-600M hidden size
            elif '300m' in self.model_name.lower() or '300' in self.model_name:
                hidden_size = 960   # ESMC-300M hidden size
            else:
                # Default to 300M size, but this should rarely happen
                hidden_size = 960
                import warnings
                warnings.warn(f"Could not determine hidden_size for model {self.model_name}, defaulting to 960. "
                            f"If this is ESMC-600M, set hidden_size to 1152 manually.")

        if self.task == 'classification':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, self.num_labels)
            )
            # Initialize classifier weights with smaller scale to prevent extreme logits
            # Using smaller initialization helps when input (pooled_repr) is normalized
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    # Use Kaiming normal with smaller gain for better stability
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    # Scale down weights to prevent extreme outputs
                    module.weight.data *= 0.1
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)

        elif self.task == 'token_classification':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, self.num_labels)
            )
            # Initialize classifier weights properly
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)

        elif self.task == 'regression':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, 1)
            )
            # Initialize classifier weights properly
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)

        # Freeze backbone if required
        # Note: Don't freeze if LoRA will be used (LoRA initialization will handle parameter freezing)
        if self.freeze_backbone and self.lora_kwargs is None:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}
    
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
            log_dict.pop("test_loss", None)
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and hasattr(self, "num_labels") and self.num_labels > 2:
            log_dict.pop("test_mcc", None)
        
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
            
            metric_name = METRIC_MAP.get(key.lower(), key)
            print(f"{metric_name}: {print_value}")
        
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
            log_dict.pop("valid_loss", None)
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and hasattr(self, "num_labels") and self.num_labels > 2:
            log_dict.pop("valid_mcc", None)
        
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
            # plt.tight_layout()
            plt.show()
