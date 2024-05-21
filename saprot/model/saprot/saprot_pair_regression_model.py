import torchmetrics
import torch
import torch.distributed as dist

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotPairRegressionModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()

        hidden_size = self.model.config.hidden_size * 2
        classifier = torch.nn.Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )

        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs_1, inputs_2):
        if self.freeze_backbone:
            hidden_1 = torch.stack(self.get_hidden_states_from_dict(inputs_1, reduction="mean"))
            hidden_2 = torch.stack(self.get_hidden_states_from_dict(inputs_2, reduction="mean"))
        else:
            # If "esm" is not in the model, use "bert" as the backbone
            backbone = self.model.esm if hasattr(self.model, "esm") else self.model.bert
            hidden_1 = backbone(**inputs_1)[0][:, 0, :]
            hidden_2 = backbone(**inputs_2)[0][:, 0, :]

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat).squeeze(dim=-1)

    def loss_func(self, stage, logits, labels):
        fitness = labels['labels'].to(logits)
        loss = torch.nn.functional.mse_loss(logits, fitness)

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(logits.detach().float(), fitness.float())

        if stage == "train":
            log_dict = self.get_log_dict("train")
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")

        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")

        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")