import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCTokenClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for ESMCBaseModel
        """
        self.num_labels = num_labels
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)

    def compute_mcc(self, preds, target):
        preds = preds.float()
        target = target.float()
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
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)

        # Tokenize sequences and obtain token ids (with special tokens handled by tokenizer)
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)

        # Determine the last layer index from the underlying backbone (handles PEFT wrapping).
        base_model = self._get_base_model()
        num_layers = getattr(base_model, "num_layers", None)

        if num_layers is not None:
            representations = self._get_representations(
                token_ids_batch,
                repr_layers=[num_layers],
            )
        else:
            representations = self._get_representations(token_ids_batch)

        pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "padding_idx", 0))
        attention_mask = (token_ids_batch != pad_token_id).unsqueeze(-1)
        representations = representations * attention_mask
        
        # Normalize representations to match HuggingFace's scale
        import torch.nn.functional as F
        representations = F.layer_norm(representations, representations.shape[-1:])

        # Directly use modules_to_save.default if it exists
        base_model = self._get_base_model()
        head = None
        
        # First, try to get modules_to_save.default directly
        if hasattr(base_model, 'classifier') and hasattr(base_model.classifier, 'modules_to_save'):
            if hasattr(base_model.classifier.modules_to_save, 'default'):
                head = base_model.classifier.modules_to_save.default
        elif hasattr(base_model, 'head') and hasattr(base_model.head, 'modules_to_save'):
            if hasattr(base_model.head.modules_to_save, 'default'):
                head = base_model.head.modules_to_save.default
        
        # Fallback to _get_head() if modules_to_save.default not found
        if head is None:
            head = self._get_head()
        
        logits = head(representations)

        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels'].to(logits.device)

        # Align label/logit lengths (ESMC embed may omit special tokens while labels keep padding)
        min_len = min(label.shape[1], logits.shape[1])
        label = label[:, :min_len]
        logits = logits[:, :min_len]

        # Flatten the logits and labels
        logits_flat = logits.view(-1, self.num_labels)
        label_flat = label.view(-1)
        
        loss = cross_entropy(logits_flat, label_flat, ignore_index=-1)

        # Remove the ignored index
        mask = label_flat != -1
        label_masked = label_flat[mask]
        logits_masked = logits_flat[mask]

        # Add the outputs to the list if not in training mode
        if stage != "train":
            preds = logits_masked.argmax(dim=-1)
            self.preds.append(preds.detach().cpu())
            self.targets.append(label_masked.detach().cpu())

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits_masked.detach(), label_masked)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

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
        self.output_test_metrics(log_dict)
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