import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
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
        print(f"Saprot Debug | device: model={model_device} self.device={self.device}")
        print(f"Saprot Debug | params: trainable={trainable_params:,} (LoRA={lora_params:,}, classifier={classifier_params:,}) / total={total_params:,} ({(trainable_params/total_params if total_params else 0):.2%})")

        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # Debug: input tensors
        if isinstance(inputs, dict):
            if 'input_ids' in inputs:
                print(f"Saprot Debug | input_ids: shape={tuple(inputs['input_ids'].shape)} dtype={inputs['input_ids'].dtype} device={inputs['input_ids'].device}")
            if 'attention_mask' in inputs:
                print(f"Saprot Debug | attn_mask: shape={tuple(inputs['attention_mask'].shape)} dtype={inputs['attention_mask'].dtype} device={inputs['attention_mask'].device}")
            # Get batch size
            if 'input_ids' in inputs and isinstance(inputs['input_ids'], torch.Tensor):
                batch_size = inputs['input_ids'].shape[0]
            elif 'attention_mask' in inputs and isinstance(inputs['attention_mask'], torch.Tensor):
                batch_size = inputs['attention_mask'].shape[0]
            else:
                batch_size = 1
            print(f"Saprot Debug | batch: {batch_size} sequences")

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            print(f"Saprot Debug | repr (after pooling): shape={tuple(repr.shape)} dtype={repr.dtype} device={repr.device}")
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)
        else:
            model_output = self.model(**inputs)
            logits = model_output.logits
            # Try to get hidden states if available
            if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
                last_hidden = model_output.hidden_states[-1]
                print(f"Saprot Debug | last_hidden: shape={tuple(last_hidden.shape)} dtype={last_hidden.dtype} device={last_hidden.device}")
        
        # Classifier stats
        try:
            if hasattr(self.model, 'classifier'):
                classifier = self.model.classifier
                cls_total = sum(p.numel() for p in classifier.parameters())
                cls_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
                print(f"Saprot Debug | classifier params: trainable={cls_trainable:,} / total={cls_total:,}")
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'classifier'):
                classifier = self.model.base_model.classifier
                cls_total = sum(p.numel() for p in classifier.parameters())
                cls_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
                print(f"Saprot Debug | classifier params: trainable={cls_trainable:,} / total={cls_total:,}")
        except Exception as e:
            print(f"Saprot Debug | classifier params: failed to get stats: {e}")
        
        print(f"Saprot Debug | logits: shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}")
        
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
        print(f"Saprot Debug | stage={stage} labels: shape={tuple(label.shape)} unique={unique}")
        print(f"Saprot Debug | stage={stage} loss: {loss.item():.6f}")

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