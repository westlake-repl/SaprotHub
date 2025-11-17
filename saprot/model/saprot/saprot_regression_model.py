import torch.distributed as dist
import torchmetrics
import torch

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
    
    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}
    
    def forward(self, inputs, structure_info=None):
        if structure_info:
            # To be implemented
            raise NotImplementedError

        # DEBUG: Get rank for distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        # For ESM models
        if hasattr(self.model, "esm"):
            # If backbone is frozen, the embedding will be the average of all residues, else it will be the
            # embedding of the <cls> token.
            if self.freeze_backbone:
                repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
                
                # DEBUG: Check repr after pooling
                if rank == 0 and hasattr(self, '_debug_step') and self._debug_step % 100 == 0:
                    print(f"[DEBUG Saprot] repr (after pooling): shape={repr.shape}, dtype={repr.dtype}, "
                          f"requires_grad={repr.requires_grad}, mean={repr.mean().item():.4f}, "
                          f"std={repr.std().item():.4f}, min={repr.min().item():.4f}, "
                          f"max={repr.max().item():.4f}")
                
                x = self.model.classifier.dropout(repr)
                x = self.model.classifier.dense(x)
                x = torch.tanh(x)
                x = self.model.classifier.dropout(x)
                logits = self.model.classifier.out_proj(x).squeeze(dim=-1)

            else:
                logits = self.model(**inputs).logits.squeeze(dim=-1)
                
                # DEBUG: Check logits from model forward
                if rank == 0 and hasattr(self, '_debug_step') and self._debug_step % 100 == 0:
                    print(f"[DEBUG Saprot] logits (from model): shape={logits.shape}, dtype={logits.dtype}, "
                          f"requires_grad={logits.requires_grad}, mean={logits.mean().item():.4f}, "
                          f"std={logits.std().item():.4f}, min={logits.min().item():.4f}, "
                          f"max={logits.max().item():.4f}")

        # For ProtBERT
        elif hasattr(self.model, "bert"):
            repr = self.model.bert(**inputs).last_hidden_state[:, 0]
            
            # DEBUG: Check repr from BERT
            if rank == 0 and hasattr(self, '_debug_step') and self._debug_step % 100 == 0:
                print(f"[DEBUG Saprot] repr (from BERT): shape={repr.shape}, dtype={repr.dtype}, "
                      f"requires_grad={repr.requires_grad}, mean={repr.mean().item():.4f}, "
                      f"std={repr.std().item():.4f}, min={repr.min().item():.4f}, "
                      f"max={repr.max().item():.4f}")
            
            logits = self.model.classifier(repr).squeeze(dim=-1)
            
            # DEBUG: Check logits from classifier
            if rank == 0 and hasattr(self, '_debug_step') and self._debug_step % 100 == 0:
                print(f"[DEBUG Saprot] logits (from classifier): shape={logits.shape}, dtype={logits.dtype}, "
                      f"requires_grad={logits.requires_grad}, mean={logits.mean().item():.4f}, "
                      f"std={logits.std().item():.4f}, min={logits.min().item():.4f}, "
                      f"max={logits.max().item():.4f}")

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        
        # DEBUG: Print diagnostic information (same as ESMC for comparison)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        
        if rank == 0 and self._debug_step % 100 == 0:
            print(f"\n[DEBUG Saprot {stage}] Step {self._debug_step}")
            print(f"  outputs: shape={outputs.shape}, dtype={outputs.dtype}, "
                  f"mean={outputs.mean().item():.6f}, std={outputs.std().item():.6f}, "
                  f"min={outputs.min().item():.6f}, max={outputs.max().item():.6f}")
            print(f"  fitness: shape={fitness.shape}, dtype={fitness.dtype}, "
                  f"mean={fitness.mean().item():.6f}, std={fitness.std().item():.6f}, "
                  f"min={fitness.min().item():.6f}, max={fitness.max().item():.6f}")
            print(f"  loss: {loss.item():.6f}")
            
            # Check if outputs and fitness are in similar ranges
            outputs_float = outputs.detach().float()
            fitness_float = fitness.float()
            print(f"  outputs (float32): mean={outputs_float.mean().item():.6f}, std={outputs_float.std().item():.6f}")
            print(f"  fitness (float32): mean={fitness_float.mean().item():.6f}, std={fitness_float.std().item():.6f}")
            
            # Check gradient flow
            if outputs.requires_grad:
                print(f"  outputs.requires_grad: True")
                if outputs.grad is not None:
                    print(f"  outputs.grad: mean={outputs.grad.mean().item():.6f}, std={outputs.grad.std().item():.6f}")
                else:
                    print(f"  outputs.grad: None (will be computed in backward)")
            else:
                print(f"  outputs.requires_grad: False (WARNING: no gradients!)")
        
        self._debug_step += 1

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)
            
            # DEBUG: Check metric values periodically
            if rank == 0 and (self._debug_step - 1) % 100 == 0 and stage == "valid":
                metric_name = [k for k, v in self.metrics[stage].items() if v is metric][0]
                try:
                    # Try to compute current metric value
                    if hasattr(metric, 'compute'):
                        metric_value = metric.compute()
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.item()
                        print(f"  {metric_name}: {metric_value:.6f}")
                except Exception as e:
                    print(f"  {metric_name}: Could not compute ({e})")
            
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