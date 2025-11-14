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
            
            # Check classifier weight gradients after loss computation (before backward)
            if debug_print:
                if hasattr(self.model, 'classifier'):
                    if hasattr(self.model.classifier, 'weight'):
                        print(f"[SAPROT] Classifier weight requires_grad: {self.model.classifier.weight.requires_grad}")
                        # Check if this weight is in optimizer
                        if hasattr(self, 'optimizer'):
                            in_optimizer = any(self.model.classifier.weight is p for group in self.optimizer.param_groups for p in group['params'])
                            print(f"[SAPROT] Classifier weight in optimizer: {in_optimizer}")
                    elif hasattr(self.model.classifier, 'out_proj') and hasattr(self.model.classifier.out_proj, 'weight'):
                        print(f"[SAPROT] Classifier out_proj weight requires_grad: {self.model.classifier.out_proj.weight.requires_grad}")
                        if hasattr(self, 'optimizer'):
                            in_optimizer = any(self.model.classifier.out_proj.weight is p for group in self.optimizer.param_groups for p in group['params'])
                            print(f"[SAPROT] Classifier out_proj weight in optimizer: {in_optimizer}")
        
        return loss
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Override to check if classifier weights are being updated"""
        # Track optimizer step calls separately (since accumulate_grad_batches may delay calls)
        if not hasattr(self, '_optimizer_step_count'):
            self._optimizer_step_count = 0
        self._optimizer_step_count += 1
        
        # Get classifier weight before optimizer step
        classifier_weight = None
        if hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'weight'):
                classifier_weight = self.model.classifier.weight
            elif hasattr(self.model.classifier, 'out_proj') and hasattr(self.model.classifier.out_proj, 'weight'):
                classifier_weight = self.model.classifier.out_proj.weight
        
        if classifier_weight is not None:
            weight_before = classifier_weight.data.clone()
            # Check gradient before optimizer step (should exist after backward)
            grad_before = classifier_weight.grad.clone() if classifier_weight.grad is not None else None
        else:
            weight_before = None
            grad_before = None
        
        # Debug print for first few optimizer steps
        debug_print = (self._optimizer_step_count <= 3)
        if debug_print and weight_before is not None:
            weight_id = id(classifier_weight)
            print(f"\n[SAPROT] ========== Before optimizer_step (optimizer call #{self._optimizer_step_count}, batch_idx={batch_idx}) ==========")
            print(f"[SAPROT]   Weight object id: {weight_id}")
            if hasattr(self, '_initial_weight_id'):
                print(f"[SAPROT]   Same weight object as initial? {weight_id == self._initial_weight_id}")
            print(f"[SAPROT]   Weight mean: {weight_before.mean().item():.6f}, std: {weight_before.std().item():.6f}")
            if grad_before is not None:
                grad_mean = grad_before.abs().mean().item()
                grad_max = grad_before.abs().max().item()
                print(f"[SAPROT]   Gradient - mean: {grad_mean:.10f}, max: {grad_max:.10f}")
            else:
                print(f"[SAPROT]   WARNING: No gradient before optimizer step!")
        
        # Call parent optimizer_step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        
        # Check if weight changed after optimizer step (only for first few steps)
        if weight_before is not None and debug_print:
            weight_after = classifier_weight.data
            weight_diff = (weight_after - weight_before).abs().mean().item()
            weight_max_diff = (weight_after - weight_before).abs().max().item()
            print(f"[SAPROT] ========== After optimizer_step (optimizer call #{self._optimizer_step_count}) ==========")
            print(f"[SAPROT]   Weight change - mean: {weight_diff:.10f}, max: {weight_max_diff:.10f}")
            if weight_diff < 1e-10:
                print(f"[SAPROT]   WARNING: Weight did NOT change after optimizer step!")
                # Check if gradient was cleared (should be None after optimizer step)
                if classifier_weight.grad is not None:
                    print(f"[SAPROT]   WARNING: Gradient still exists after optimizer step (should be None)")
                else:
                    print(f"[SAPROT]   Gradient cleared (normal after optimizer step)")
            else:
                print(f"[SAPROT]   ✓ Weight updated successfully")
                print(f"[SAPROT]   New weight mean: {weight_after.mean().item():.6f}, std: {weight_after.std().item():.6f}")
    
    def on_train_epoch_end(self):
        """Print classifier weight changes at the end of each training epoch"""
        super().on_train_epoch_end()
        
        # Get classifier weight
        classifier_weight = None
        actual_classifier = None
        if hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'weight'):
                actual_classifier = self.model.classifier
                classifier_weight = self.model.classifier.weight
            elif hasattr(self.model.classifier, 'out_proj') and hasattr(self.model.classifier.out_proj, 'weight'):
                actual_classifier = self.model.classifier.out_proj
                classifier_weight = self.model.classifier.out_proj.weight
        
        # Store initial weights if not already stored
        if not hasattr(self, '_initial_weight_std') and classifier_weight is not None:
            self._initial_weight_std = classifier_weight.std().item()
            self._initial_weight_mean = classifier_weight.mean().item()
            self._initial_weight = classifier_weight.data.clone()
            # Store weight object id to verify we're checking the same weight
            self._initial_weight_id = id(classifier_weight)
            print(f"[SAPROT] Storing initial weight for tracking:")
            print(f"[SAPROT]   - Weight object id: {self._initial_weight_id}")
            print(f"[SAPROT]   - Weight mean: {self._initial_weight_mean:.6f}, std: {self._initial_weight_std:.6f}")
            print(f"[SAPROT]   - Classifier type: {type(actual_classifier)}")
        
        # Print current weight stats
        if classifier_weight is not None:
            current_mean = classifier_weight.mean().item()
            current_std = classifier_weight.std().item()
            epoch = getattr(self, 'epoch', 0)
            current_weight_id = id(classifier_weight)
            
            print(f"\n[SAPROT] ========== Epoch {epoch} End - Classifier Weight Stats ==========")
            print(f"[SAPROT] Classifier type: {type(actual_classifier)}")
            print(f"[SAPROT] Weight object id: {current_weight_id}")
            if hasattr(self, '_initial_weight_id'):
                print(f"[SAPROT] Same weight object as initial? {current_weight_id == self._initial_weight_id}")
            print(f"[SAPROT] Current weight - mean: {current_mean:.6f}, std: {current_std:.6f}")
            
            if hasattr(self, '_initial_weight_std'):
                mean_change = current_mean - self._initial_weight_mean
                std_change = current_std - self._initial_weight_std
                std_change_pct = (std_change / self._initial_weight_std) * 100 if self._initial_weight_std > 0 else 0
                print(f"[SAPROT] Change from initial - mean: {mean_change:+.6f}, std: {std_change:+.6f} ({std_change_pct:+.2f}%)")
                
                # Check actual weight difference (more reliable than mean/std)
                if hasattr(self, '_initial_weight'):
                    weight_diff = (classifier_weight.data - self._initial_weight).abs().mean().item()
                    weight_max_diff = (classifier_weight.data - self._initial_weight).abs().max().item()
                    print(f"[SAPROT] Weight absolute difference - mean: {weight_diff:.8f}, max: {weight_max_diff:.8f}")
            
            # Verify this weight is in the optimizer
            if hasattr(self, 'optimizer') and classifier_weight is not None:
                in_optimizer = False
                optimizer_lr = None
                for group in self.optimizer.param_groups:
                    if any(classifier_weight is p for p in group['params']):
                        in_optimizer = True
                        optimizer_lr = group.get('lr', None)
                        break
                print(f"[SAPROT] Weight in optimizer: {in_optimizer}")
                if optimizer_lr is not None:
                    print(f"[SAPROT] Optimizer learning rate for this weight: {optimizer_lr}")
            
            print(f"[SAPROT] ================================================================\n")
        else:
            epoch = getattr(self, 'epoch', 0)
            print(f"\n[SAPROT] ========== Epoch {epoch} End - Classifier Weight Stats ==========")
            print(f"[SAPROT] WARNING: Could not find classifier weight to check!")
            print(f"[SAPROT] ================================================================\n")
    
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