import torchmetrics
import torch
import torch.nn.functional as F

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ESMCBaseModel


@register_model
class ESMCPairClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels

            **kwargs: other arguments for ESMCBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="pair_classification", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs_1, inputs_2):
        proteins_1 = self._parse_proteins_input(inputs_1)
        proteins_2 = self._parse_proteins_input(inputs_2)

        token_ids_1, attention_mask_1, tokenizer = self._tokenize_sequences(proteins_1)
        token_ids_2, attention_mask_2, _ = self._tokenize_sequences(proteins_2)

        # Forward through model (will automatically use LoRA if PEFT is applied)
        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            model_output_1 = self.model(
                input_ids=token_ids_1,
                attention_mask=attention_mask_1
            )
            reps_1 = model_output_1.last_hidden_state
            
            model_output_2 = self.model(
                input_ids=token_ids_2,
                attention_mask=attention_mask_2
            )
            reps_2 = model_output_2.last_hidden_state

        reps_1 = F.layer_norm(reps_1, reps_1.shape[-1:])
        reps_2 = F.layer_norm(reps_2, reps_2.shape[-1:])

        mask_1 = attention_mask_1.unsqueeze(-1)
        mask_2 = attention_mask_2.unsqueeze(-1)

        h1 = (reps_1 * mask_1).sum(dim=1) / mask_1.sum(dim=1).clamp(min=1)
        h2 = (reps_2 * mask_2).sum(dim=1) / mask_2.sum(dim=1).clamp(min=1)

        hidden_concat = torch.cat([h1, h2], dim=-1)
        head = self._get_head()
        return head(hidden_concat)

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