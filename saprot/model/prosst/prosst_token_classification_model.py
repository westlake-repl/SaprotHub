import torch
import torchmetrics
from torch.nn.functional import cross_entropy

from ..model_interface import register_model
from .base import ProSSTBaseModel

try:
    from saprot.data.prosst_labels import RESIDUE_LABEL_IGNORE_INDEX
except ImportError:
    from data.prosst_labels import RESIDUE_LABEL_IGNORE_INDEX


@register_model
class ProSSTTokenClassificationModel(ProSSTBaseModel):
    def __init__(self, num_labels: int, test_result_path: str = None, **kwargs):
        self.num_labels = num_labels
        self.test_result_path = test_result_path
        super().__init__(task="token_classification", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        hidden_size = self.model.config.hidden_size
        classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_labels),
        )
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs):
        representations = self.get_token_representations(inputs)
        return self._get_classifier()(representations)

    def loss_func(self, stage, logits, labels):
        label = labels["labels"]
        if logits.shape[:2] != label.shape:
            raise ValueError(
                "Residue-level logits and labels must have matching batch/token "
                f"dimensions: logits={tuple(logits.shape[:2])}, "
                f"labels={tuple(label.shape)}."
            )
        flat_logits = logits.reshape(-1, self.num_labels)
        flat_labels = label.reshape(-1)
        labeled_mask = flat_labels != RESIDUE_LABEL_IGNORE_INDEX
        if not torch.any(labeled_mask):
            raise ValueError("A residue-level batch must contain a labeled residue.")

        labeled_logits = flat_logits[labeled_mask]
        labeled_targets = flat_labels[labeled_mask]
        loss = cross_entropy(labeled_logits, labeled_targets)

        for metric in self.metrics[stage].values():
            metric.update(labeled_logits.detach(), labeled_targets)

        if stage == "test" and self.test_result_path is not None:
            for sample_logits, sample_labels in zip(logits, label):
                sample_mask = sample_labels != RESIDUE_LABEL_IGNORE_INDEX
                token_positions = torch.nonzero(sample_mask).flatten()
                self.test_token_outputs.append(
                    (
                        token_positions.detach().cpu(),
                        sample_logits[sample_mask].detach().cpu(),
                        sample_labels[sample_mask].detach().cpu(),
                    )
                )

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_token_outputs = []

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            with open(self.test_result_path, "w", encoding="utf-8") as handle:
                headers = ["sample_index", "residue_index", "pred", "target"]
                headers.extend([f"prob_{idx}" for idx in range(self.num_labels)])
                handle.write(",".join(headers) + "\n")

                for sample_index, (positions, logits, targets) in enumerate(
                    self.test_token_outputs
                ):
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = probabilities.argmax(dim=-1)
                    for position, pred, target, probs in zip(
                        positions,
                        predictions,
                        targets,
                        probabilities,
                    ):
                        row = [
                            str(sample_index),
                            str(position.item()),
                            str(pred.item()),
                            str(target.item()),
                        ]
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
