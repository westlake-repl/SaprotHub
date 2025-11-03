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
                 **kwargs):
        assert task in ["classification", "token_classification", "regression"]
        self.task = task
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing

        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        super().__init__(**kwargs)

        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []

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


