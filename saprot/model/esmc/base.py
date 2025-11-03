import torch

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

        print(f"[ESMC][LoRA] Injected LoRA into {replaced} Linear layers. r={cfg['r']} alpha={cfg['lora_alpha']} dropout={cfg['lora_dropout']}")
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
        print(f"[ESMC][LoRA] Trainable params: {trainable} / {total} ({trainable/total:.2%})")

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
            print("[ESMC] ESMProtein.__init__ signature:", inspect.signature(ESMProtein.__init__))
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


