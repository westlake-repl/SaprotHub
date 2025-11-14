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
                 extra_config: dict = None,
                 **kwargs):
        """
        Args:
            task: Task name.

            model_name: Model name for ESMC (esmc_300m or esmc_600m)

            device: Device to use (cuda or cpu)

            freeze_backbone: Whether to freeze the backbone of the model

            gradient_checkpointing: Whether to enable gradient checkpointing

            lora_kwargs: LoRA configuration

            extra_config: Extra config for the model (for compatibility, not used by ESMC)

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ["classification", "token_classification", "regression", "pair_classification", "pair_regression"]
        self.task = task
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs
        self.extra_config = extra_config  # Store for compatibility, though ESMC doesn't use it
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
        """Initialize LoRA adapters"""
        from peft import LoraConfig, get_peft_model, PeftModel
        import os
        
        is_trainable = getattr(self.lora_kwargs, "is_trainable", False)
        config_list = getattr(self.lora_kwargs, "config_list", [])
        assert self.lora_kwargs.num_lora >= len(config_list), ("The number of LoRA models should be greater than or "
                                                               "equal to the number of weight files.")
        for i in range(self.lora_kwargs.num_lora):
            adapter_name = f"adapter_{i}" if self.lora_kwargs.num_lora > 1 else "default"
            
            # Load pre-trained LoRA weights
            if i < len(config_list):
                entry = config_list[i]
                lora_config_path = getattr(entry, "lora_config_path", entry)
                
                if not isinstance(lora_config_path, str):
                    # Assume a Peft config object was provided directly
                    if i == 0:
                        self.model = get_peft_model(self.model, lora_config_path, adapter_name=adapter_name)
                    else:
                        self.model.load_adapter(lora_config_path, adapter_name=adapter_name, is_trainable=is_trainable)
                    continue

                if not os.path.exists(lora_config_path):
                    raise FileNotFoundError(f"LoRA config path '{lora_config_path}' does not exist.")

                if i == 0:
                    # Initialize with pre-trained LoRA weights
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_config_path,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
                else:
                    self.model.load_adapter(
                        lora_config_path,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
            
            # Initialize LoRA model for training
            else:
                # Get user-specified target_modules, or use default + find additional layers
                target_modules = getattr(self.lora_kwargs, "target_modules", None)
                
                if target_modules is None:
                    # Cache the found target_modules to avoid re-computation
                    if not hasattr(self, '_cached_target_modules'):
                        # Base ESMC patterns (equivalent to SaProt's 5 modules)
                        base_targets = ["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"]
                        
                        # Find all linear layers to discover additional layers
                        all_linear_layers = []
                        for name, module in self.model.named_modules():
                            if isinstance(module, torch.nn.Linear) and 'embed' not in name.lower():
                                all_linear_layers.append(name)
                        
                        # Start with base targets
                        matched_layers = set(base_targets)
                        
                        # Find additional FFN layers (ffn.0, ffn.2, etc.) if they exist
                        for layer_name in all_linear_layers:
                            parts = layer_name.split('.')
                            for part in parts:
                                # Match any ffn.X pattern
                                if 'ffn.' in part:
                                    # Extract ffn.X pattern
                                    if part.startswith('ffn.') and part[4:].replace('.', '').isdigit():
                                        matched_layers.add(part)
                        
                        # Try to find additional attention components
                        keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "proj", "gate", "up", "down"]
                        for layer_name in all_linear_layers:
                            parts = layer_name.split('.')
                            for part in parts:
                                for keyword in keywords:
                                    if keyword in part.lower() and part not in matched_layers:
                                        # Only add if it's a meaningful layer (not already covered)
                                        if 'attn' in layer_name.lower() or 'ffn' in layer_name.lower():
                                            matched_layers.add(part)
                                            break
                        
                        # Convert to sorted list for consistency
                        self._cached_target_modules = sorted(list(matched_layers))
                    
                    target_modules = self._cached_target_modules
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
            
        self.model.print_trainable_parameters()
        
        # After LoRA model is initialized, add trainable parameters to optimizer)
        self.init_optimizers()
    
    def initialize_model(self):
        """Initialize ESMC model and task-specific classifiers"""
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
        self.model = ESMC.from_pretrained(self.model_name)
        self.model = self.model.to(torch.float32)
        self.tokenizer = self.model.tokenizer

        # Attach simple heads per task
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

        # Task-specific classifier initialization
        if self.task == 'classification':
            # Match HuggingFace AutoModelForSequenceClassification structure:
            # Dropout -> Dense -> Tanh -> Dropout -> OutProj
            classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),  # First dropout (before dense)
                torch.nn.Linear(hidden_size, hidden_size),  # Dense layer
                torch.nn.Tanh(),  # Tanh activation
                torch.nn.Dropout(0.1),  # Second dropout (after tanh)
                torch.nn.Linear(hidden_size, self.num_labels)  # Output projection
            )
            # Initialize classifier weights properly (Xavier for Tanh activation)
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    # Use Xavier uniform for Tanh activation (matches HuggingFace ESM models)
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)
            # Also expose as a generic head for naming consistency
            setattr(self.model, "head", classifier)

        elif self.task == 'token_classification':
            # Match HuggingFace AutoModelForTokenClassification structure:
            # Simple Linear layer (no intermediate layers, no dropout, no activation)
            classifier = torch.nn.Linear(hidden_size, self.num_labels)
            # Initialize classifier weights properly (match HuggingFace initialization)
            # HuggingFace uses very small initialization for classifier (typically 0.002-0.005)
            # This ensures initial logits are close to zero, resulting in low initial loss
            torch.nn.init.normal_(classifier.weight, mean=0.0, std=0.002)
            if classifier.bias is not None:
                torch.nn.init.zeros_(classifier.bias)
            setattr(self.model, "classifier", classifier)
            setattr(self.model, "head", classifier)

        elif self.task == 'regression':
            # Match HuggingFace AutoModelForSequenceClassification structure:
            # Dropout -> Dense -> Tanh -> Dropout -> OutProj
            classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),  # First dropout (before dense)
                torch.nn.Linear(hidden_size, hidden_size),  # Dense layer
                torch.nn.Tanh(),  # Tanh activation
                torch.nn.Dropout(0.1),  # Second dropout (after tanh)
                torch.nn.Linear(hidden_size, 1)  # Output projection
            )
            # Initialize classifier weights properly (Xavier for Tanh activation)
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)
            setattr(self.model, "head", classifier)

        elif self.task == 'pair_classification':
            pair_hidden = hidden_size * 2
            classifier = torch.nn.Sequential(
                torch.nn.Linear(pair_hidden, pair_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(pair_hidden, self.num_labels)
            )
            for module in classifier:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "classifier", classifier)
            setattr(self.model, "head", classifier)

        elif self.task == 'pair_regression':
            pair_hidden = hidden_size * 2
            reg_head = torch.nn.Sequential(
                torch.nn.Linear(pair_hidden, pair_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(pair_hidden, 1)
            )
            for module in reg_head:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            setattr(self.model, "reg_head", reg_head)
            setattr(self.model, "head", reg_head)
            setattr(self.model, "classifier", reg_head)

        # Freeze backbone if required
        if self.freeze_backbone and self.lora_kwargs is None:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}
    
    # Generic Helper Methods
    def _parse_proteins_input(self, inputs):
        """
        Parse proteins input from inputs dict

        Args:
            inputs: Input dict containing 'proteins' key
            
        Returns:
            proteins: List of ESMProtein objects
        """
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError(f"{self.__class__.__name__}.forward expects inputs['proteins'] (list of ESMProtein)")
        if not isinstance(proteins, list):
            proteins = [proteins]
        return proteins
    
    def _get_tokenizer(self):
        """
        Get tokenizer from model, handling PEFT wrapping

        Returns:
            tokenizer: The tokenizer object
        """
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'tokenizer'):
            return self.model.base_model.model.tokenizer
        elif hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        else:
            raise AttributeError("Cannot find tokenizer in model")
    
    def _get_head(self):
        """
        Get task head module (prefer `head`, fallback to `classifier`), handling PEFT wrapping
        
        Returns:
            head: The task head module
        """
        # PEFT-wrapped path first
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            base = self.model.base_model.model
            if hasattr(base, 'head'):
                return base.head
            if hasattr(base, 'classifier'):
                return base.classifier
        # Non-PEFT
        if hasattr(self.model, 'head'):
            return self.model.head
        if hasattr(self.model, 'classifier'):
            return self.model.classifier
        raise AttributeError("Cannot find task head (head/classifier) in model")

    # Backward-compatible alias
    def _get_classifier(self):
        return self._get_head()
    
    def _get_base_model(self):
        """
        Get base model from PEFT-wrapped model or return model itself
        
        Returns:
            base_model: The base model (unwrapped from PEFT if needed)
        """
        if hasattr(self.model, 'base_model'):
            # PEFT wrapped model: get the actual base model
            return self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
        else:
            # Not wrapped by PEFT: return model directly
            return self.model
    
    # Tokenization & Forward Methods
    def _tokenize_sequences(self, proteins, return_tensors="pt", device=None):
        """
        Tokenize protein sequences and pad them
        
        Args:
            proteins: List of ESMProtein objects
            return_tensors: Return format ("pt" for PyTorch tensors)
            device: Device to move tensors to. If None, uses self.device
            
        Returns:
            token_ids_batch: Token IDs tensor [B, L]
            attention_mask: Attention mask tensor [B, L]
            tokenizer: The tokenizer used
        """
        sequences = [p.sequence for p in proteins]
        tokenizer = self._get_tokenizer()
        
        batch_encoding = tokenizer(
            sequences, 
            padding=True, 
            return_tensors=return_tensors
        )
        
        if device is None:
            device = self.device
        
        token_ids_batch = batch_encoding['input_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)
        
        return token_ids_batch, attention_mask, tokenizer
    
    def _forward_backbone(self, token_ids_batch, repr_layers=None):
        """
        Forward pass through backbone, handling PEFT wrapping
        Note: This method does NOT handle freeze_backbone context - 
        callers should wrap this in appropriate context if needed.
        
        Args:
            token_ids_batch: Token IDs tensor [B, L]
            repr_layers: List of layer indices to return representations from (optional)
            
        Returns:
            model_output: Model output containing hidden_states or representations
        """
        base_model = self._get_base_model()
        
        # When wrapped by PEFT, forward may receive kwargs, but ESMC expects positional args
        if repr_layers is not None:
            # For models that use repr_layers parameter
            model_output = base_model.forward(token_ids_batch, repr_layers=repr_layers)
        else:
            # Standard forward
            model_output = base_model.forward(token_ids_batch)
        
        return model_output
    
    def _get_representations(self, token_ids_batch, repr_layers=None, layer_idx=-1):
        """
        Forward through backbone and extract representations, handling freeze_backbone
        This wraps the forward pass in appropriate gradient context based on freeze_backbone flag.
        
        Args:
            token_ids_batch: Token IDs tensor [B, L]
            repr_layers: List of layer indices to return representations from (optional)
            layer_idx: Which layer to extract from hidden_states. -1 means last layer.
                      If model uses repr_layers, this is used to index into representations dict.
            
        Returns:
            representations: Hidden states tensor [B, L, D]
        """
        # Debug print
        debug_print = (hasattr(self, '_debug_step') and self._debug_step <= 3)
        if debug_print:
            print(f"[ESMC] _get_representations: freeze_backbone={self.freeze_backbone}")
            print(f"[ESMC] _get_representations: torch.is_grad_enabled()={torch.is_grad_enabled()}")
        
        # Wrap backbone forward in no_grad context when freeze_backbone=True
        # This saves memory and computation when backbone is frozen
        if self.freeze_backbone:
            if debug_print:
                print(f"[ESMC] _get_representations: Using torch.no_grad()")
            with torch.no_grad():
                model_output = self._forward_backbone(token_ids_batch, repr_layers=repr_layers)
        else:
            if debug_print:
                print(f"[ESMC] _get_representations: Using torch.enable_grad()")
            with torch.enable_grad():
                model_output = self._forward_backbone(token_ids_batch, repr_layers=repr_layers)
            
        # Extract representations based on output format
        if repr_layers is not None:
            # Model returns dict with 'representations' key
            representations = model_output['representations'][repr_layers[0]]
        else:
            # Model returns object with hidden_states attribute
            representations = model_output.hidden_states[layer_idx]
        
        if debug_print:
            print(f"[ESMC] _get_representations: representations.requires_grad={representations.requires_grad}")
            print(f"[ESMC] _get_representations: After context, torch.is_grad_enabled()={torch.is_grad_enabled()}")
        
        return representations
    
    # Pooling & Normalization Methods
    def _pool_representations(self, representations, token_ids_batch, pad_token_id=None):
        """
        Pool sequence representations by averaging over sequence length (excluding padding).
        
        Args:
            representations: Hidden states tensor [B, L, D]
            token_ids_batch: Token IDs tensor [B, L] for creating mask
            pad_token_id: Padding token ID. If None, will try to get from tokenizer or model
            
        Returns:
            pooled_repr: Pooled representation tensor [B, D]
        """
        # Get pad_token_id if not provided
        if pad_token_id is None:
            tokenizer = self._get_tokenizer()
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else getattr(tokenizer, 'padding_idx', 0)
        
        # Create mask and compute sequence lengths
        mask = (token_ids_batch != pad_token_id).unsqueeze(-1)
        sequence_lengths = mask.sum(dim=1)
        sequence_lengths = sequence_lengths.clamp(min=1)
        
        # Average pooling over sequence length
        pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths
        
        return pooled_repr
    
    def _normalize_pooled_repr(self, pooled_repr):
        """
        Normalize pooled representation to prevent extreme values.
        This is useful for classification tasks to improve numerical stability.
        
        Args:
            pooled_repr: Pooled representation tensor [B, D]
            
        Returns:
            normalized_repr: Normalized pooled representation tensor [B, D]
        """
        pooled_mean = pooled_repr.mean(dim=-1, keepdim=True)
        pooled_std = pooled_repr.std(dim=-1, keepdim=True)
        pooled_std = pooled_std.clamp(min=1e-6)
        normalized_repr = (pooled_repr - pooled_mean) / pooled_std
        
        return normalized_repr

    def get_hidden_states_from_seqs(self, seqs: list, reduction: str = None) -> list:
        """
        Get hidden representations of protein sequences (通用 - 所有任务).
        
        Args:
            seqs: A list of protein sequences (amino acid sequences as strings)
            reduction: Whether to reduce the hidden states. If None, returns sequence-level representations for each position.
                       If "mean", the hidden states are averaged over the sequence length.
        
        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [D] if reduction="mean", 
                          or [L, D] if reduction=None, where L is the sequence length and D is the hidden dimension.
        """
        # Get tokenizer - handle PEFT wrapping
        tokenizer = self._get_tokenizer()
        
        # Tokenize sequences
        batch_encoding = tokenizer(
            seqs, 
            padding=True, 
            return_tensors="pt"
        )
        
        # Get device - use device_str from initialization or infer from model
        if hasattr(self, 'device_str'):
            device = torch.device(self.device_str)
        elif hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        token_ids_batch = batch_encoding['input_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)
        
        # Forward pass to get representations
        with torch.no_grad():
            model_output = self._forward_backbone(token_ids_batch)
            representations = model_output.hidden_states[-1]  # [B, L, D]
        
        # Process each sequence
        repr_list = []
        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.padding_idx
        
        for i in range(len(seqs)):
            # Get mask for this sequence (excluding padding tokens)
            mask = (token_ids_batch[i] != pad_token_id)
            
            if reduction == "mean":
                # Average pooling over sequence length (excluding padding)
                seq_repr = representations[i][mask].mean(dim=0)  # [D]
            else:
                # Return all hidden states for this sequence (excluding padding)
                seq_repr = representations[i][mask]  # [L, D]
            
            repr_list.append(seq_repr)
        
        return repr_list
    
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        Rewrite this function to save LoRA parameters (通用 - 所有任务).
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
            "valid_loss": "Root mean squared error (RMSE)", # Only for regression task
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
