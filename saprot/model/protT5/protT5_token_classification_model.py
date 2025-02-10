import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ProtT5BaseModel


@register_model
class ProtT5TokenClassificationModel(ProtT5BaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)

    def initialize_model(self):
        super().initialize_model()

        hidden_size = self.model.config.hidden_size
        classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_labels)
        )

        setattr(self.model, "classifier", classifier)
    
    def compute_mcc(self, preds, target):
        tp = (preds * target).sum()
        tn = ((1 - preds) * (1 - target)).sum()
        fp = (preds * (1 - target)).sum()
        fn = ((1 - preds) * target).sum()
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        return tp, tn, fp, fn, mcc
    
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}
    
    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # Get the input_ids and attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)
        
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state 
            logits = self.model.classifier(sequence_output)

        return logits[:]
    
    
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        # Flatten the logits and labels
        logits = logits.view(-1, self.num_labels)
        label = label.view(-1)

        # Remove the ignored index
        mask = label != 0
        label = label[mask]
        logits = logits[mask]

        loss = cross_entropy(logits, label, ignore_index=0)
        
        
        
        # Add the outputs to the list if not in training mode
        if stage != "train":
            preds = logits.argmax(dim=-1)
            self.preds.append(preds)
            self.targets.append(label)
        
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

        
        preds = torch.cat(self.preds, dim=-1)
        target = torch.cat(self.targets, dim=-1)
        tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
        
        # Gather results
        # tmp = torch.tensor([tp, tn, fp, fn])
        # tp, tn, fp, fn = self.all_gather(tmp).sum(dim=0)
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        log_dict["test_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        # if dist.get_rank() == 0:
        #     print(log_dict)
        print('='*100)
        print('Test Result:')
        for key, value in log_dict.items():
            print(f"{key}: {value.item()}")
        print('='*100)
        self.log_info(log_dict)
        self.reset_metrics("test")
    
    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        preds = torch.cat(self.preds, dim=-1)
        target = torch.cat(self.targets, dim=-1)
        tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
        
        # Gather results
        # tmp = torch.tensor([tp, tn, fp, fn])
        # tp, tn, fp, fn = self.all_gather(tmp).sum(dim=0)
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        log_dict["valid_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)