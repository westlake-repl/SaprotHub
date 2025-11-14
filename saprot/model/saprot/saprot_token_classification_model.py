import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotTokenClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)
    
    def compute_mcc(self, preds, target):
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
        # ========== SAPROT DEBUG: Forward Start ==========
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1
        debug_print = (self._debug_step <= 3)  # Only print first 3 steps
        
        if debug_print:
            print(f"\n[SAPROT DEBUG Step {self._debug_step}] ========== Forward ==========")
            print(f"[SAPROT] freeze_backbone: {self.freeze_backbone}")
            if isinstance(inputs, dict):
                print(f"[SAPROT] inputs keys: {list(inputs.keys())}")
                if 'input_ids' in inputs:
                    print(f"[SAPROT] input_ids shape: {inputs['input_ids'].shape}")
                    print(f"[SAPROT] input_ids sample (first 10): {inputs['input_ids'][0, :10].tolist() if inputs['input_ids'].shape[0] > 0 else 'N/A'}")
                if 'attention_mask' in inputs:
                    print(f"[SAPROT] attention_mask shape: {inputs['attention_mask'].shape}")
        
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            if debug_print:
                print(f"[SAPROT] freeze_backbone=True path")
                print(f"[SAPROT] repr shape: {repr.shape}")
                print(f"[SAPROT] repr mean: {repr.mean().item():.6f}, std: {repr.std().item():.6f}")
                print(f"[SAPROT] repr requires_grad: {repr.requires_grad}")
            
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            if debug_print:
                print(f"[SAPROT] After dense - x mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
            
            x = torch.tanh(x)
            if debug_print:
                print(f"[SAPROT] After tanh - x mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
            
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)
        
        else:
            if debug_print:
                print(f"[SAPROT] freeze_backbone=False path - using model(**inputs).logits")
            
            logits = self.model(**inputs).logits
            
            if debug_print:
                # Check if we can access intermediate representations
                if hasattr(self.model, 'esm'):
                    print(f"[SAPROT] Using ESM backbone")
                elif hasattr(self.model, 'bert'):
                    print(f"[SAPROT] Using BERT backbone")
        
        if debug_print:
            print(f"[SAPROT] logits shape: {logits.shape}")
            print(f"[SAPROT] logits mean: {logits.mean().item():.6f}")
            print(f"[SAPROT] logits std: {logits.std().item():.6f}")
            print(f"[SAPROT] logits min: {logits.min().item():.6f}")
            print(f"[SAPROT] logits max: {logits.max().item():.6f}")
            print(f"[SAPROT] logits requires_grad: {logits.requires_grad}")
            print(f"[SAPROT] logits sample (first token, all classes): {logits[0, 0, :].tolist() if logits.shape[0] > 0 and logits.shape[1] > 0 else 'N/A'}")
            # Check classifier structure
            if hasattr(self.model, 'classifier'):
                if hasattr(self.model.classifier, 'weight'):
                    print(f"[SAPROT] Classifier is Linear - weight mean: {self.model.classifier.weight.mean().item():.6f}, std: {self.model.classifier.weight.std().item():.6f}")
                    print(f"[SAPROT] Classifier weight requires_grad: {self.model.classifier.weight.requires_grad}")
                elif hasattr(self.model.classifier, 'out_proj'):
                    print(f"[SAPROT] Classifier has out_proj - weight mean: {self.model.classifier.out_proj.weight.mean().item():.6f}, std: {self.model.classifier.out_proj.weight.std().item():.6f}")
            print(f"[SAPROT] ========== Forward End ==========\n")
        
        return logits[:]
    
    def loss_func(self, stage, logits, labels):
        debug_print = (hasattr(self, '_debug_step') and self._debug_step <= 3)
        
        label = labels['labels']
        
        if debug_print:
            print(f"\n[SAPROT DEBUG Step {self._debug_step}] ========== Loss Function ==========")
            print(f"[SAPROT] stage: {stage}")
            print(f"[SAPROT] Original label shape: {label.shape}")
            print(f"[SAPROT] Original logits shape: {logits.shape}")
            print(f"[SAPROT] Label unique values: {torch.unique(label).tolist()}")
            print(f"[SAPROT] Label distribution: {torch.bincount(label[label >= 0].long() + 1).tolist() if (label >= 0).any() else 'N/A'}")
        
        # Flatten the logits and labels
        logits_flat = logits.view(-1, self.num_labels)
        label_flat = label.view(-1)
        
        if debug_print:
            print(f"[SAPROT] Flattened - label_flat shape: {label_flat.shape}, logits_flat shape: {logits_flat.shape}")
            print(f"[SAPROT] logits_flat mean: {logits_flat.mean().item():.6f}, std: {logits_flat.std().item():.6f}")
        
        loss = cross_entropy(logits_flat, label_flat, ignore_index=-1)
        
        if debug_print:
            print(f"[SAPROT] Loss (before mask): {loss.item():.6f}")
        
        # Remove the ignored index
        mask = label_flat != -1
        label_masked = label_flat[mask]
        logits_masked = logits_flat[mask]
        
        if debug_print:
            print(f"[SAPROT] After removing ignore_index - valid samples: {mask.sum().item()}/{len(mask)}")
            print(f"[SAPROT] label_masked unique values: {torch.unique(label_masked).tolist() if len(label_masked) > 0 else 'N/A'}")
            print(f"[SAPROT] label_masked distribution: {torch.bincount(label_masked.long()).tolist() if len(label_masked) > 0 else 'N/A'}")
            print(f"[SAPROT] logits_masked mean: {logits_masked.mean().item():.6f}, std: {logits_masked.std().item():.6f}")
            # Check predictions
            preds_masked = logits_masked.argmax(dim=-1)
            print(f"[SAPROT] preds_masked unique values: {torch.unique(preds_masked).tolist() if len(preds_masked) > 0 else 'N/A'}")
            print(f"[SAPROT] preds_masked distribution: {torch.bincount(preds_masked).tolist() if len(preds_masked) > 0 else 'N/A'}")
            # Check accuracy
            if len(label_masked) > 0:
                acc = (preds_masked == label_masked).float().mean().item()
                print(f"[SAPROT] Accuracy (on this batch): {acc:.6f}")
            print(f"[SAPROT] Final loss: {loss.item():.6f}")
            # Check gradients
            if logits.requires_grad:
                print(f"[SAPROT] logits requires_grad: True")
                if logits.grad is not None:
                    print(f"[SAPROT] logits.grad mean: {logits.grad.mean().item():.6f}")
            else:
                print(f"[SAPROT] logits requires_grad: False")
            print(f"[SAPROT] ========== Loss Function End ==========\n")
        
        # Add the outputs to the list if not in training mode
        if stage != "train":
            preds = logits_masked.argmax(dim=-1)
            self.preds.append(preds)
            self.targets.append(label_masked)
        
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