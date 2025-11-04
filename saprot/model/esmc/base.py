import torch
import os
import matplotlib.pyplot as plt

from ..abstract_model import AbstractModel

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

        # Align order with saprot: apply LoRA right after model init, then set metrics list
        if self.lora_kwargs is not None:
            from easydict import EasyDict
            # No need to freeze backbone if LoRA is used
            self.freeze_backbone = False
            self.lora_kwargs = EasyDict(self.lora_kwargs)
            # Use PEFT-based LoRA to align with Saprot
            self._init_lora_peft()
            # Re-apply freezing logic after LoRA initialization to ensure correctness
            # This is important because subclass may override classifier after LoRA init
            self._apply_lora_freezing()
            self.init_optimizers()

        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []

    # (removed lightweight injector per user's request)

    def _init_lora_peft(self):
        """
        Apply LoRA via PEFT to ESMC backbone, mirroring Saprot's usage.
        """
        from peft import LoraConfig, get_peft_model
        # Defaults similar to Saprot/your Colab example; user config can override
        r = getattr(self.lora_kwargs, "r", 8)
        lora_alpha = getattr(self.lora_kwargs, "lora_alpha", 16)
        lora_dropout = getattr(self.lora_kwargs, "lora_dropout", 0.0)
        target_modules = getattr(self.lora_kwargs, "target_modules", [
            "q_proj", "k_proj", "v_proj", "out_proj", "fc", "mlp",
            "intermediate.dense", "output.dense"
        ])
        task_type = "FEATURE_EXTRACTION"
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=task_type,
        )
        # Wrap model with PEFT
        self.model = get_peft_model(self.model, config)
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass
        # Ensure classifier is trainable
        if hasattr(self.model, 'classifier'):
            for n, p in self.model.classifier.named_parameters():
                p.requires_grad = True
        # Freeze non-LoRA, non-classifier params (align with Saprot behavior: only update LoRA+head)
        for n, p in self.model.named_parameters():
            is_lora = ("lora_" in n)
            is_classifier = n.startswith("classifier")
            p.requires_grad = is_lora or is_classifier
    
    def _apply_lora_freezing(self):
        """
        Apply freezing logic after LoRA initialization.
        This should be called after LoRA is initialized and classifier is created.
        """
        # First, freeze ALL parameters
        for n, p in self.model.named_parameters():
            p.requires_grad = False

        # Then, unfreeze only LoRA parameters and classifier
        for n, p in self.model.named_parameters():
            is_lora = ("lora_" in n)
            is_classifier = n.startswith("classifier")
            if is_lora or is_classifier:
                p.requires_grad = True

        # Ensure classifier is trainable (double-check)
        if hasattr(self.model, 'classifier'):
            for name, param in self.model.classifier.named_parameters():
                param.requires_grad = True

        # Debug: print parameter status
        total, trainable = 0, 0
        lora_params = 0
        classifier_params = 0
        for n, p in self.model.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                if "lora_" in n:
                    lora_params += num
                elif n.startswith("classifier"):
                    classifier_params += num
        print(f"ESMC LoRA Freezing Applied: Trainable={trainable:,} (LoRA={lora_params:,}, Classifier={classifier_params:,}) / Total={total:,} ({trainable/total:.2%})")

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

        # Print ESMProtein constructor signature for debugging compatibility
        try:
            import inspect
            from esm.sdk.api import ESMProtein
        except Exception as _e:
            print("ESMC Failed to inspect ESMProtein signature:", _e)

        # Attach simple heads per task
        hidden_size = getattr(getattr(self.model, 'config', None), 'hidden_size', 960)

        if self.task == 'classification':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, self.num_labels)
            )
            setattr(self.model, "classifier", classifier)

        elif self.task == 'token_classification':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, self.num_labels)
            )
            setattr(self.model, "classifier", classifier)

        elif self.task == 'regression':
            classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_size, 1)
            )
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
        Rewrite this function to save LoRA parameters (for ESMC lightweight LoRA)
        """
        
        if not self.lora_kwargs:
            # If not using LoRA, use default save method
            return super().save_checkpoint(save_path, save_info, save_weights_only)
        
        else:
            # If PEFT is active, prefer saving adapter via save_pretrained
            try:
                if hasattr(self.model, 'save_pretrained'):
                    dir = os.path.dirname(save_path)
                    os.makedirs(dir, exist_ok=True)
                    self.model.save_pretrained(dir)
                    print(f"ESMC LoRA (PEFT) adapter saved to {dir}")
                    return
            except Exception:
                pass
            # Fallback: Save lightweight LoRA parameters (A and B) and classifier
            try:
                if hasattr(self.trainer.strategy, "deepspeed_engine"):
                    save_path = os.path.dirname(save_path)
            except Exception as e:
                pass
            
            dir = os.path.dirname(save_path)
            os.makedirs(dir, exist_ok=True)
            
            state_dict = {} if save_info is None else save_info
            
            # Extract only LoRA parameters (A and B) and classifier
            lora_state_dict = {}
            for name, param in self.model.named_parameters():
                # Save LoRA parameters (A and B) and classifier
                if ("A" in name or "B" in name) or name.startswith("classifier"):
                    lora_state_dict[name] = param.float()  # Convert to fp32
            
            state_dict["model"] = lora_state_dict
            
            if not save_weights_only:
                state_dict["global_step"] = self.step
                state_dict["epoch"] = self.epoch
                state_dict["best_value"] = getattr(self, "best_value", None)
                state_dict["lr_scheduler"] = self.lr_schedulers().state_dict()
                
                # If not using DeepSpeed, save optimizer state
                try:
                    if not hasattr(self.trainer.strategy, "deepspeed_engine"):
                        state_dict["optimizer"] = self.optimizers().optimizer.state_dict()
                except Exception:
                    pass
            
            torch.save(state_dict, save_path)
            print(f"ESMC LoRA checkpoint saved to {save_path}")

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
            try:
                from google.colab import widgets
                width = 400 * len(log_dict)
                height = 400
                self.grid = widgets.Grid(1, 1, header_row=False, header_column=False,
                                         style=f'width:{width}px; height:{height}px')
            except ImportError:
                # If not in Colab, create a simple grid alternative
                self.grid = None
        
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
        
        if self.grid is not None:
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
                    ax[idx].set_title(METRIC_MAP.get(metric.lower(), metric.upper()))
                    ax[idx].set_xlabel('step')
                    ax[idx].set_ylabel(METRIC_MAP.get(metric.lower(), metric))
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
        else:
            # Fallback for non-Colab environments
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
                ax[idx].set_title(METRIC_MAP.get(metric.lower(), metric.upper()))
                ax[idx].set_xlabel('step')
                ax[idx].set_ylabel(METRIC_MAP.get(metric.lower(), metric))
                ax[idx].plot(self.valid_metrics_list['step'], self.valid_metrics_list[metric], marker='o')
            
            plt.tight_layout()
            # plt.tight_layout()
            plt.show()



    # Ensure optimizer only sees LoRA + classifier as trainable
    def configure_optimizers(self):
        try:
            self._apply_lora_freezing()
        except Exception:
            pass
        return super().configure_optimizers()