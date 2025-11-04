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
        # Let the base class load the ESMC model and set matmul precision
        super().initialize_model()
        
        # Overwrite the placeholder classifier from the base class with the correct, multi-layer one
        embed_dim = self.model.embed.embedding_dim
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, self.num_labels)
        )
        
        # All freezing/unfreezing logic is now handled correctly in the base class.
        # No need to manually set requires_grad here.

        # This debug print is still useful to confirm module names if needed in the future.
        try:
            import torch.nn as nn
            print("--- ESMC Linear Module Names (for LoRA target_modules) ---")
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    print(f"  {n}")
            print("---------------------------------------------------------")
        except Exception as _e:
            print("ESMC Debug | listing Linear modules failed:", _e)

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        # Tokenization & Padding
        sequences = [p.sequence for p in proteins]
        batch_encoding = self.model.tokenizer(
            sequences, 
            padding=True, 
            return_tensors="pt"
        )
        token_ids_batch = batch_encoding['input_ids'].to(self.device)
        attention_mask = batch_encoding['attention_mask'].to(self.device)

        # Inference - PEFT model forward pass is the same
        # The 'with torch.no_grad()' context is removed as it's handled by Lightning and requires_grad flags
        model_output = self.model.forward(token_ids_batch)
        representations = model_output.hidden_states[-1]

        # --- IMPROVED & SIMPLIFIED POOLING ---
        # Use the attention_mask from the tokenizer for robust mean pooling.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(representations.size()).float()
        sum_embeddings = torch.sum(representations * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_repr = sum_embeddings / sum_mask

        # Pass through the classifier
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
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")
        self.plot_valid_metrics_curve(log_dict)