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
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for ESMCBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        # Wrap backbone forward and pooling in no_grad context when freeze_backbone=True
        # This saves memory and computation when backbone is frozen
        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # Get base model for embed and padding_idx (handle PEFT wrapping)
            if hasattr(self.model, 'base_model'):
                base_model = self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
            else:
                base_model = self.model
            
            # Tokenize and pad using ESMProtein.embed
            token_ids_list = [base_model.embed(p) for p in proteins]
            token_ids_batch = pad_sequence(token_ids_list, batch_first=True, padding_value=base_model.padding_idx)
            token_ids_batch = token_ids_batch.to(self.device)
            
            # Forward pass to get representations
            # When wrapped by PEFT, forward may receive kwargs, but ESMC expects positional args
            if hasattr(self.model, 'base_model'):
                model_output = base_model.forward(token_ids_batch)
            else:
                model_output = self.model.forward(token_ids_batch)
            representations = model_output.hidden_states[-1]

            # Pooling
            mask = (token_ids_batch != base_model.padding_idx).unsqueeze(-1)
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths
            
            # Normalize pooled representation to prevent extreme values
            pooled_mean = pooled_repr.mean(dim=-1, keepdim=True)
            pooled_std = pooled_repr.std(dim=-1, keepdim=True)
            pooled_std = pooled_std.clamp(min=1e-6)
            pooled_repr = (pooled_repr - pooled_mean) / pooled_std

        # Classifier always needs gradients (even when backbone is frozen)
        # Get classifier - handle PEFT wrapping
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'classifier'):
            classifier = self.model.base_model.model.classifier
        elif hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
        else:
            raise AttributeError("Cannot find classifier in model")
        
        logits = classifier(pooled_repr)
        
        # Convert to float32 for numerical stability
        logits = logits.float()
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-20.0, max=20.0)

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