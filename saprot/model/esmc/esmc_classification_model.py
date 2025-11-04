import torchmetrics
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence

from saprot.model.model_interface import register_model
from saprot.model.esmc.base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)}

    def initialize_model(self):
        super().initialize_model()
        
        # Debug: list all Linear module names to help configure lora_kwargs.target_modules
        try:
            import torch.nn as nn
            print("ESMC Debug | Linear module names (for target_modules):")
            count = 0
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    print(f"  {n}")
                    count += 1
            print(f"ESMC Debug | Total Linear modules: {count}")
        except Exception as _e:
            print("ESMC Debug | listing Linear modules failed:", _e)

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        # Debug: devices and param stats
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = getattr(self, 'device', 'cpu')
        total_params = 0
        trainable_params = 0
        lora_params = 0
        classifier_params = 0
        for n, p in self.model.named_parameters():
            num = p.numel()
            total_params += num
            if p.requires_grad:
                trainable_params += num
                # Count LoRA params from either lightweight (A/B) or PEFT (lora_*)
                if (
                    n.endswith('.A') or n.endswith('.B') or '.A.' in n or '.B.' in n
                    or ('lora_' in n)
                ):
                    lora_params += num
                elif 'classifier' in n:
                    # PEFT wraps model, so classifier path may be 'base_model.model.classifier.xxx'
                    # or just 'classifier.xxx' - check for substring instead of startswith
                    classifier_params += num
        print(f"ESMC Debug | device: model={model_device} self.device={self.device}")
        print(f"ESMC Debug | params: trainable={trainable_params:,} (LoRA={lora_params:,}, classifier={classifier_params:,}) / total={total_params:,} ({(trainable_params/total_params if total_params else 0):.2%})")

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Tokenization & Padding
            sequences = [p.sequence for p in proteins]
            print(f"ESMC Debug | batch: {len(sequences)} sequences; lens={[len(s) for s in sequences][:8]}{'...' if len(sequences)>8 else ''}")
            # Get tokenizer - PEFT usually keeps it accessible at top level, but check base_model if needed
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'tokenizer'):
                tokenizer = self.model.base_model.model.tokenizer
            elif hasattr(self.model, 'tokenizer'):
                tokenizer = self.model.tokenizer
            else:
                raise AttributeError("Cannot find tokenizer in model")
            batch_encoding = tokenizer(
                sequences, 
                padding=True, 
                return_tensors="pt"
            )
            token_ids_batch = batch_encoding['input_ids'].to(self.device)
            attention_mask = batch_encoding['attention_mask'].to(self.device)
            print(f"ESMC Debug | token_ids: shape={tuple(token_ids_batch.shape)} dtype={token_ids_batch.dtype} device={token_ids_batch.device}")
            print(f"ESMC Debug | attn_mask: shape={tuple(attention_mask.shape)} dtype={attention_mask.dtype} device={attention_mask.device}")

            # Inference
            # When wrapped by PEFT, forward may receive kwargs, but ESMC expects positional args
            # Access base_model directly if PEFT is active
            if hasattr(self.model, 'base_model'):
                # PEFT wrapped model: call base_model forward with positional args
                base_model = self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
                model_output = base_model.forward(token_ids_batch)
            else:
                # Not wrapped by PEFT: direct call
                model_output = self.model.forward(token_ids_batch)
            representations = model_output.hidden_states[-1]
            print(f"ESMC Debug | last_hidden: shape={tuple(representations.shape)} dtype={representations.dtype} device={representations.device}")

            #  Pooling
            # Get tokenizer from base_model if PEFT wrapped, otherwise from model directly
            if hasattr(self.model, 'base_model'):
                tokenizer = self.model.base_model.model.tokenizer if hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'tokenizer') else self.model.tokenizer
            else:
                tokenizer = self.model.tokenizer
            mask = (token_ids_batch != tokenizer.pad_token_id).unsqueeze(-1)
            
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths
            print(f"ESMC Debug | pooled: shape={tuple(pooled_repr.shape)} dtype={pooled_repr.dtype} device={pooled_repr.device}")

        # Get classifier - handle PEFT wrapping
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'classifier'):
            classifier = self.model.base_model.model.classifier
        elif hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
        else:
            raise AttributeError("Cannot find classifier in model")
        
        logits = classifier(pooled_repr)
        # Classifier stats
        cls_total = sum(p.numel() for p in classifier.parameters())
        cls_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(f"ESMC Debug | classifier params: trainable={cls_trainable:,} / total={cls_total:,}")
        print(f"ESMC Debug | logits: shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}")

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        # Convert logits to float32 for stable loss calculation (FP16 can cause numerical instability)
        logits = logits.float()
        loss = cross_entropy(logits, label)

        # Debug loss and label
        try:
            unique = torch.unique(label).tolist()
        except Exception:
            unique = []
        print(f"ESMC Debug | stage={stage} labels: shape={tuple(label.shape)} unique={unique}")
        print(f"ESMC Debug | stage={stage} loss: {loss.item():.6f}")

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)