import torch
import torchmetrics

from ..model_interface import register_model
from .base import ProSSTBaseModel


@register_model
class ProSSTRegressionModel(ProSSTBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        hidden_size = self.model.config.hidden_size
        classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {
            f"{stage}_loss": torchmetrics.MeanSquaredError(),
            f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
            f"{stage}_R2": torchmetrics.R2Score(),
            f"{stage}_pearson": torchmetrics.PearsonCorrCoef(),
        }

    def forward(self, inputs):
        pooled = self.get_pooled_representations(inputs)
        return self._get_classifier()(pooled).squeeze(dim=-1)

    def loss_func(self, stage, outputs, labels):
        targets = labels["labels"].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        for metric in self.metrics[stage].values():
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), targets)

        if stage == "test" and self.test_result_path is not None:
            if not hasattr(self, "test_preds"):
                self.test_preds = []
                self.test_targets = []
            self.test_preds.append(outputs.detach().cpu())
            self.test_targets.append(targets.detach().cpu())

        if stage == "train":
            self.log_info({"train_loss": loss.item()})
            self.reset_metrics("train")

        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            preds = torch.cat(self.test_preds, dim=0)
            targets = torch.cat(self.test_targets, dim=0)

            with open(self.test_result_path, "w", encoding="utf-8") as handle:
                handle.write("pred,target\n")
                for pred, target in zip(preds, targets):
                    handle.write(f"{pred.item()},{target.item()}\n")

        log_dict = self.get_log_dict("test")
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        self.plot_valid_metrics_curve(log_dict)
