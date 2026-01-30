import torchmetrics
import torch.distributed as dist
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCRegressionModel(ESMCBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for ESMCBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs, coords=None):
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)

        # Tokenization & Padding
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)

        # Backbone representations
        representations = self._get_representations(token_ids_batch)

        # Normalize representations before pooling (mirrors pair_regression behaviour)
        representations = F.layer_norm(representations, representations.shape[-1:])

        # Pooling - always needs gradients for head training
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)

        # CRITICAL FIX: Directly use modules_to_save.default if it exists
        # This ensures we use the same weight object that's being trained
        base_model = self._get_base_model()

        # Fallback to _get_head()
        head = self._get_head()

        # using tanh not sigmoid
        pooled_repr = torch.tanh(pooled_repr)
        logits = head(pooled_repr).squeeze(dim=-1)
        
        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)

        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors
            
            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))
            
            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            if dist.get_rank() == 0:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        w.write(f"{pred.item()}\t{target.item()}\n")
        
        log_dict = self.get_log_dict("test")

        # if dist.get_rank() == 0:
        #     print(log_dict)

        self.output_test_metrics(log_dict)

        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        
        self.plot_valid_metrics_curve(log_dict)
