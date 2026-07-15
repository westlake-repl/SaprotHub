import torch
import torchmetrics
from torch.nn.functional import cross_entropy

from ..model_interface import register_model
from .pair_base import ProSSTPairBaseModel


@register_model
class ProSSTPairClassificationModel(ProSSTPairBaseModel):
    def __init__(self, num_labels: int, test_result_path: str = None, **kwargs):
        self.num_labels = num_labels
        self.test_result_path = test_result_path
        super().__init__(
            task="pair_classification",
            output_size=num_labels,
            **kwargs,
        )

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def loss_func(self, stage, logits, labels):
        targets = labels["labels"]
        loss = cross_entropy(logits, targets)
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), targets)

        if stage == "test" and self.test_result_path is not None:
            self.test_logits.append(logits.detach().cpu())
            self.test_labels.append(targets.detach().cpu())
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")
        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_logits = []
        self.test_labels = []

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            logits = torch.cat(self.test_logits, dim=0)
            targets = torch.cat(self.test_labels, dim=0)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            with open(self.test_result_path, "w", encoding="utf-8") as handle:
                headers = ["pred", "target"]
                headers.extend(f"prob_{idx}" for idx in range(self.num_labels))
                handle.write(",".join(headers) + "\n")
                for pred, target, probs in zip(
                    predictions,
                    targets,
                    probabilities,
                ):
                    row = [str(pred.item()), str(target.item())]
                    row.extend(str(prob.item()) for prob in probs)
                    handle.write(",".join(row) + "\n")

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
