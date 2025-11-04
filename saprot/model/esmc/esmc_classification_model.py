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
        
        embed_dim = self.model.embed.embedding_dim
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, self.num_labels)
        )
        
        # If LoRA is used, re-apply freezing logic after classifier is created
        # Note: LoRA initialization happens after initialize_model() in __init__,
        # so we need to apply freezing logic after LoRA is initialized
        # This is handled by _apply_lora_freezing() in base class after LoRA init
        # But we also apply it here as a safeguard
        if self.lora_kwargs is not None:
            # Check if LoRA has been initialized (by checking if LoRA parameters exist)
            has_lora_params = any("A" in n or "B" in n for n, _ in self.model.named_parameters())
            if has_lora_params:
                # LoRA already initialized, apply freezing logic
                if hasattr(self, '_apply_lora_freezing'):
                    self._apply_lora_freezing()
            # If LoRA not yet initialized, freezing will be applied in __init__ after LoRA init
        else:
            # If not using LoRA, ensure classifier is trainable
            for name, param in self.model.named_parameters():
                if name.startswith("classifier"):
                    param.requires_grad = True

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Tokenization & Padding
            sequences = [p.sequence for p in proteins]
            batch_encoding = self.model.tokenizer(
                sequences, 
                padding=True, 
                return_tensors="pt"
            )
            token_ids_batch = batch_encoding['input_ids'].to(self.device)
            attention_mask = batch_encoding['attention_mask'].to(self.device)

            # Inference
            model_output = self.model.forward(token_ids_batch)
            representations = model_output.hidden_states[-1]

            #  Pooling
            mask = (token_ids_batch != self.model.tokenizer.pad_token_id).unsqueeze(-1)
            
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths

        logits = self.model.classifier(pooled_repr)

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        
        loss = cross_entropy(logits, label)

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