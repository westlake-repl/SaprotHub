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

        # For ESM models
        if hasattr(self.model, "esm"):
            print(f"[DEBUG][Saprot::tokenize] input_ids_shape={inputs['input_ids'].shape}, "
                  f"attention_mask_shape={inputs['attention_mask'].shape}, device={inputs['input_ids'].device}")
            
            # If backbone is frozen, the embedding will be the average of all residues, else it will be the
            # embedding of the <cls> token.
            if self.freeze_backbone:
                repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
                print(f"[DEBUG][Saprot::representations] freeze_backbone=True, repr_shape={repr.shape}, "
                      f"mean={repr.mean().item():.6f}, std={repr.std().item():.6f}, "
                      f"min={repr.min().item():.6f}, max={repr.max().item():.6f}")
                
                x = self.model.classifier.dropout(repr)
                print(f"[DEBUG][Saprot::classifier] After Dropout: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
                
                x = self.model.classifier.dense(x)
                print(f"[DEBUG][Saprot::classifier] After Dense: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
                      f"min={x.min().item():.6f}, max={x.max().item():.6f}")
                
                x = torch.tanh(x)
                print(f"[DEBUG][Saprot::classifier] After Tanh: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
                      f"min={x.min().item():.6f}, max={x.max().item():.6f}")
                
                x = self.model.classifier.dropout(x)
                print(f"[DEBUG][Saprot::classifier] After Dropout2: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
                
                logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
                print(f"[DEBUG][Saprot::forward:end] freeze_backbone=True, logits_shape={logits.shape}, "
                      f"mean={logits.mean().item():.6f}, std={logits.std().item():.6f}, "
                      f"min={logits.min().item():.6f}, max={logits.max().item():.6f}")

            else:
                # Use AutoModelForSequenceClassification's forward which uses CLS token
                model_output = self.model(**inputs)
                logits = model_output.logits.squeeze(dim=-1)
                print(f"[DEBUG][Saprot::forward:end] freeze_backbone=False, logits_shape={logits.shape}, "
                      f"mean={logits.mean().item():.6f}, std={logits.std().item():.6f}, "
                      f"min={logits.min().item():.6f}, max={logits.max().item():.6f}")
                print(f"[DEBUG][Saprot::classifier] Using AutoModelForSequenceClassification forward (CLS token)")

        # For ProtBERT
        elif hasattr(self.model, "bert"):
            repr = self.model.bert(**inputs).last_hidden_state[:, 0]
            print(f"[DEBUG][Saprot::representations] ProtBERT, repr_shape={repr.shape}, "
                  f"mean={repr.mean().item():.6f}, std={repr.std().item():.6f}")
            logits = self.model.classifier(repr).squeeze(dim=-1)
            print(f"[DEBUG][Saprot::forward:end] ProtBERT, logits_shape={logits.shape}, "
                  f"mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        print(f"[DEBUG][Saprot::loss:{stage}] outputs_mean={outputs.mean().item():.6f}, outputs_std={outputs.std().item():.6f}, "
              f"outputs_min={outputs.min().item():.6f}, outputs_max={outputs.max().item():.6f}, "
              f"labels_mean={fitness.mean().item():.6f}, labels_std={fitness.std().item():.6f}, "
              f"labels_min={fitness.min().item():.6f}, labels_max={fitness.max().item():.6f}")
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        print(f"[DEBUG][Saprot::loss:{stage}] loss={loss.item():.6f}")
        
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