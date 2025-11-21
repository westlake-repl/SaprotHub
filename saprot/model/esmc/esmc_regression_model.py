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
        print(f"[DEBUG][ESMC::tokenize] token_ids_shape={token_ids_batch.shape}, attention_mask_shape={attention_mask.shape}, device={token_ids_batch.device}")

        # Backbone representations
        representations = self._get_representations(token_ids_batch)
        print(f"[DEBUG][ESMC::representations] reps_shape={representations.shape}, dtype={representations.dtype}, "
              f"mean={representations.mean().item():.6f}, std={representations.std().item():.6f}, "
              f"min={representations.min().item():.6f}, max={representations.max().item():.6f}")

        # Use mean pooling like ESMC classification does
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)
        print(f"[DEBUG][ESMC::pooled] pooled_shape={pooled_repr.shape}, "
              f"mean={pooled_repr.mean().item():.6f}, std={pooled_repr.std().item():.6f}, "
              f"min={pooled_repr.min().item():.6f}, max={pooled_repr.max().item():.6f}")

        # Normalize pooled representations to stabilize the regression head
        normalized_pooled_repr = self._normalize_pooled_repr(pooled_repr)
        print(f"[DEBUG][ESMC::pooled_norm] shape={normalized_pooled_repr.shape}, "
              f"mean={normalized_pooled_repr.mean().item():.6f}, std={normalized_pooled_repr.std(unbiased=False).item():.6f}, "
              f"min={normalized_pooled_repr.min().item():.6f}, max={normalized_pooled_repr.max().item():.6f}")

        # CRITICAL FIX: Directly use modules_to_save.default if it exists
        # This ensures we use the same weight object that's being trained
        base_model = self._get_base_model()
        head = None
        
        # First, try to get modules_to_save.default directly
        if hasattr(base_model, 'classifier') and hasattr(base_model.classifier, 'modules_to_save'):
            if hasattr(base_model.classifier.modules_to_save, 'default'):
                head = base_model.classifier.modules_to_save.default
                print(f"[DEBUG][ESMC::head] Found head from base_model.classifier.modules_to_save.default")
        elif hasattr(base_model, 'head') and hasattr(base_model.head, 'modules_to_save'):
            if hasattr(base_model.head.modules_to_save, 'default'):
                head = base_model.head.modules_to_save.default
                print(f"[DEBUG][ESMC::head] Found head from base_model.head.modules_to_save.default")
        
        # Fallback to _get_head() if modules_to_save.default not found
        if head is None:
            head = self._get_head()
            print(f"[DEBUG][ESMC::head] Using _get_head() fallback")
        
        # Pass through classifier layers step by step for debugging
        head_input = normalized_pooled_repr

        if isinstance(head, torch.nn.Sequential):
            x = head[0](head_input)  # Dropout
            print(f"[DEBUG][ESMC::classifier] After Dropout: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            
            x = head[1](x)  # Linear
            print(f"[DEBUG][ESMC::classifier] After Linear1: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
                  f"min={x.min().item():.6f}, max={x.max().item():.6f}")
            
            x = head[2](x)  # Tanh
            print(f"[DEBUG][ESMC::classifier] After Tanh: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
                  f"min={x.min().item():.6f}, max={x.max().item():.6f}")
            
            x = head[3](x)  # Dropout
            print(f"[DEBUG][ESMC::classifier] After Dropout2: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            
            logits = head[4](x).squeeze(dim=-1)  # Linear
        else:
            # If head is not Sequential, just call it directly
            logits = head(head_input).squeeze(dim=-1)
        
        print(f"[DEBUG][ESMC::forward:end] logits_shape={logits.shape}, "
              f"mean={logits.mean().item():.6f}, std={logits.std(unbiased=False).item():.6f}, "
              f"min={logits.min().item():.6f}, max={logits.max().item():.6f}")

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        print(f"[DEBUG][ESMC::loss:{stage}] outputs_mean={outputs.mean().item():.6f}, outputs_std={outputs.std(unbiased=False).item():.6f}, "
              f"outputs_min={outputs.min().item():.6f}, outputs_max={outputs.max().item():.6f}, "
              f"labels_mean={fitness.mean().item():.6f}, labels_std={fitness.std(unbiased=False).item():.6f}, "
              f"labels_min={fitness.min().item():.6f}, labels_max={fitness.max().item():.6f}")
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        print(f"[DEBUG][ESMC::loss:{stage}] loss={loss.item():.6f}")

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