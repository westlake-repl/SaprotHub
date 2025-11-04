import torch
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
            self._init_lora_lightweight(hidden_size=None)
            self.init_optimizers()

        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []

    # ------------------------- LoRA for ESMC (lightweight) -------------------------
    def _init_lora_lightweight(self, hidden_size: int) -> None:
        """
        A minimal LoRA injection for ESMC: replace selected Linear layers with LoRALinear.
        Defaults target to common attention/ffn linear layers; keep classifier trainable.
        """
        import types
        import torch.nn as nn
        from typing import Iterable

        class LoRALinear(nn.Module):
            def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
                super().__init__()
                self.in_features = base_linear.in_features
                self.out_features = base_linear.out_features
                self.r = r
                self.scaling = alpha / float(r) if r > 0 else 0.0
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                # Freeze base weight/bias
                self.weight = base_linear.weight
                self.bias = base_linear.bias
                self.weight.requires_grad_(False)
                if self.bias is not None:
                    self.bias.requires_grad_(False)
                # LoRA params
                self.A = nn.Parameter(torch.zeros(self.in_features, r)) if r > 0 else None
                self.B = nn.Parameter(torch.zeros(r, self.out_features)) if r > 0 else None
                self.reset_parameters()

            def reset_parameters(self):
                if self.A is not None and self.B is not None:
                    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                    nn.init.zeros_(self.B)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.nn.functional.linear(x, self.weight, self.bias)
                if self.A is not None and self.B is not None:
                    lora = self.dropout(x) @ self.A @ self.B
                    out = out + self.scaling * lora
                return out

        import math
        import torch.nn as nn

        cfg = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": [
                # common ESMC linear names (best-effort):
                "attn.layernorm_qkv.1",  # qkv projection linear inside Sequential
                "attn.out_proj",
                "ffn.1",                  # first FFN linear
                "ffn.3",                  # second FFN linear
            ],
            **(self.lora_kwargs or {}),
        }

        def name_matches(name: str, targets: Iterable[str]) -> bool:
            for t in targets:
                if name.endswith(t):
                    return True
            return False

        replaced = 0
        for name, module in list(self.model.named_modules()):
            # skip classifier
            if name.endswith("classifier"):
                continue
            if isinstance(module, nn.Linear) and name_matches(name, cfg["target_modules"]):
                # find parent module and attribute
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.split('.')[-1]
                parent = self.model if parent_name == '' else dict(self.model.named_modules())[parent_name]
                lora_layer = LoRALinear(module, r=cfg["r"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"])
                setattr(parent, attr_name, lora_layer)
                replaced += 1

        print(f"ESMC LoRA: Injected LoRA into {replaced} Linear layers. r={cfg['r']} alpha={cfg['lora_alpha']} dropout={cfg['lora_dropout']}")
        # Freeze all backbone params except LoRA and classifier
        for n, p in self.model.named_parameters():
            allow = ("A" in n or "B" in n) or n.startswith("classifier")
            p.requires_grad = allow
        # report
        total, trainable = 0, 0
        for p in self.model.parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        print(f"ESMC LoRA: Trainable params: {trainable} / {total} ({trainable/total:.2%})")

    def initialize_model(self) -> None:
        # Workaround: ESMC's EsmSequenceTokenizer defines special tokens as read-only
        # properties that clash with transformers setters in some versions.
        # Remove those class-level properties before tokenizer construction.
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
        self.model = ESMC.from_pretrained(self.model_name).to(self.device_str)
        self.tokenizer = self.model.tokenizer

        # Print ESMProtein constructor signature for debugging compatibility
        try:
            import inspect
            from esm.sdk.api import ESMProtein
        except Exception as _e:
            print("[ESMC] Failed to inspect ESMProtein signature:", _e)

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
        if self.freeze_backbone:
            for p in self.model.embed.parameters():
                p.requires_grad = False
            for p in self.model.transformer.parameters():
                p.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}

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


