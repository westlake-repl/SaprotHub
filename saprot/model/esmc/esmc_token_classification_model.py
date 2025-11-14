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
        # ========== ESMC DEBUG: Forward Start ==========
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1
        debug_print = (self._debug_step <= 3)  # Only print first 3 steps
        
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)
        if debug_print:
            print(f"\n[ESMC DEBUG Step {self._debug_step}] ========== Forward ==========")
            print(f"[ESMC] Number of proteins: {len(proteins)}")
            if len(proteins) > 0:
                print(f"[ESMC] First protein sequence length: {len(proteins[0].sequence)}")

        # Tokenize sequences and obtain token ids (with special tokens handled by tokenizer)
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)
        if debug_print:
            print(f"[ESMC] token_ids_batch shape: {token_ids_batch.shape}")
            print(f"[ESMC] attention_mask shape: {attention_mask.shape}")
            print(f"[ESMC] token_ids_batch sample (first 10): {token_ids_batch[0, :10].tolist() if token_ids_batch.shape[0] > 0 else 'N/A'}")

        # Determine the last layer index from the underlying backbone (handles PEFT wrapping).
        base_model = self._get_base_model()
        num_layers = getattr(base_model, "num_layers", None)
        if debug_print:
            print(f"[ESMC] freeze_backbone: {self.freeze_backbone}")
            print(f"[ESMC] num_layers: {num_layers}")

        if num_layers is not None:
            representations = self._get_representations(
                token_ids_batch,
                repr_layers=[num_layers],
            )
        else:
            representations = self._get_representations(token_ids_batch)
        
        if debug_print:
            print(f"[ESMC] representations shape: {representations.shape}")
            print(f"[ESMC] representations mean: {representations.mean().item():.6f}")
            print(f"[ESMC] representations std: {representations.std().item():.6f}")
            print(f"[ESMC] representations min: {representations.min().item():.6f}")
            print(f"[ESMC] representations max: {representations.max().item():.6f}")
            print(f"[ESMC] representations requires_grad: {representations.requires_grad}")

        pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "padding_idx", 0))
        attention_mask = (token_ids_batch != pad_token_id).unsqueeze(-1)
        representations = representations * attention_mask
        
        # Normalize representations to match HuggingFace's scale
        # HuggingFace models typically have normalized hidden states before classifier
        # This allows us to use std=0.02 for classifier weights (matching HuggingFace)
        # and get reasonable logits scale (std ~0.1-0.2 like Saprot)
        import torch.nn.functional as F
        representations = F.layer_norm(representations, representations.shape[-1:])
        
        if debug_print:
            print(f"[ESMC] After masking - representations mean: {representations.mean().item():.6f}")
            print(f"[ESMC] After masking - representations std: {representations.std().item():.6f}")
            print(f"[ESMC] After LayerNorm - representations mean: {representations.mean().item():.6f}")
            print(f"[ESMC] After LayerNorm - representations std: {representations.std().item():.6f}")

        # Head always needs gradients
        head = self._get_head()
        if debug_print:
            # Check which classifier is actually being used
            print(f"[ESMC] Head type: {type(head)}")
            print(f"[ESMC] Head module path: {head}")
            
            # Also check base model structure
            base_model = self._get_base_model()
            if hasattr(base_model, 'classifier'):
                print(f"[ESMC] Base model has classifier: {type(base_model.classifier)}")
                if hasattr(base_model.classifier, 'modules_to_save'):
                    print(f"[ESMC] Base classifier has modules_to_save")
                    if hasattr(base_model.classifier.modules_to_save, 'default'):
                        print(f"[ESMC] Base classifier.modules_to_save.default: {type(base_model.classifier.modules_to_save.default)}")
            
            # Check if it's a LoRA-wrapped classifier
            if hasattr(head, 'original_module'):
                print(f"[ESMC] Using LoRA-wrapped classifier (original_module)")
                actual_head = head.original_module
            elif hasattr(head, 'modules_to_save'):
                print(f"[ESMC] Using classifier with modules_to_save")
                # Check if we should use modules_to_save.default instead
                if hasattr(head, 'modules_to_save') and hasattr(head.modules_to_save, 'default'):
                    print(f"[ESMC] Found modules_to_save.default, checking which one is used...")
            else:
                actual_head = head
            
            # Check classifier weights
            if hasattr(head, 'weight'):
                print(f"[ESMC] Classifier weight shape: {head.weight.shape}")
                print(f"[ESMC] Classifier weight mean: {head.weight.mean().item():.6f}")
                print(f"[ESMC] Classifier weight std: {head.weight.std().item():.6f}")
                print(f"[ESMC] Classifier weight requires_grad: {head.weight.requires_grad}")
                # Verify this weight is in optimizer
                if hasattr(self, 'optimizer'):
                    in_optimizer = any(head.weight is p for group in self.optimizer.param_groups for p in group['params'])
                    print(f"[ESMC] Classifier weight in optimizer: {in_optimizer}")
                # Check if this is the actual weight being used
                if hasattr(head, 'original_module'):
                    orig_weight = head.original_module.weight
                    print(f"[ESMC] Original module weight mean: {orig_weight.mean().item():.6f}, std: {orig_weight.std().item():.6f}")
            elif hasattr(head, 'original_module') and hasattr(head.original_module, 'weight'):
                orig_weight = head.original_module.weight
                print(f"[ESMC] Using original_module.weight - mean: {orig_weight.mean().item():.6f}, std: {orig_weight.std().item():.6f}")
            elif hasattr(head, '__getitem__'):  # Sequential or other indexable
                for i in range(len(head)):
                    if hasattr(head[i], 'weight'):
                        print(f"[ESMC] Classifier[{i}] weight mean: {head[i].weight.mean().item():.6f}, std: {head[i].weight.std().item():.6f}")
                        print(f"[ESMC] Classifier[{i}] weight requires_grad: {head[i].weight.requires_grad}")
        
        logits = head(representations)
        
        if debug_print:
            print(f"[ESMC] logits shape: {logits.shape}")
            print(f"[ESMC] logits mean: {logits.mean().item():.6f}")
            print(f"[ESMC] logits std: {logits.std().item():.6f}")
            print(f"[ESMC] logits min: {logits.min().item():.6f}")
            print(f"[ESMC] logits max: {logits.max().item():.6f}")
            print(f"[ESMC] logits sample (first token, all classes): {logits[0, 0, :].tolist() if logits.shape[0] > 0 and logits.shape[1] > 0 else 'N/A'}")
            print(f"[ESMC] ========== Forward End ==========\n")

        return logits

    def loss_func(self, stage, logits, labels):
        debug_print = (hasattr(self, '_debug_step') and self._debug_step <= 3)
        
        label = labels['labels'].to(logits.device)
        
        if debug_print:
            print(f"\n[ESMC DEBUG Step {self._debug_step}] ========== Loss Function ==========")
            print(f"[ESMC] stage: {stage}")
            print(f"[ESMC] Original label shape: {label.shape}")
            print(f"[ESMC] Original logits shape: {logits.shape}")
            print(f"[ESMC] Label unique values: {torch.unique(label).tolist()}")
            print(f"[ESMC] Label distribution: {torch.bincount(label[label >= 0].long() + 1).tolist() if (label >= 0).any() else 'N/A'}")

        # Align label/logit lengths (ESMC embed may omit special tokens while labels keep padding)
        min_len = min(label.shape[1], logits.shape[1])
        label = label[:, :min_len]
        logits = logits[:, :min_len]
        
        if debug_print:
            print(f"[ESMC] After alignment - label shape: {label.shape}, logits shape: {logits.shape}")

        # Flatten the logits and labels
        logits_flat = logits.view(-1, self.num_labels)
        label_flat = label.view(-1)
        
        if debug_print:
            print(f"[ESMC] Flattened - label_flat shape: {label_flat.shape}, logits_flat shape: {logits_flat.shape}")
            print(f"[ESMC] logits_flat mean: {logits_flat.mean().item():.6f}, std: {logits_flat.std().item():.6f}")
        
        loss = cross_entropy(logits_flat, label_flat, ignore_index=-1)
        
        if debug_print:
            print(f"[ESMC] Loss (before mask): {loss.item():.6f}")

        # Remove the ignored index
        mask = label_flat != -1
        label_masked = label_flat[mask]
        logits_masked = logits_flat[mask]
        
        if debug_print:
            print(f"[ESMC] After removing ignore_index - valid samples: {mask.sum().item()}/{len(mask)}")
            print(f"[ESMC] label_masked unique values: {torch.unique(label_masked).tolist() if len(label_masked) > 0 else 'N/A'}")
            print(f"[ESMC] label_masked distribution: {torch.bincount(label_masked.long()).tolist() if len(label_masked) > 0 else 'N/A'}")
            print(f"[ESMC] logits_masked mean: {logits_masked.mean().item():.6f}, std: {logits_masked.std().item():.6f}")
            # Check predictions
            preds_masked = logits_masked.argmax(dim=-1)
            print(f"[ESMC] preds_masked unique values: {torch.unique(preds_masked).tolist() if len(preds_masked) > 0 else 'N/A'}")
            print(f"[ESMC] preds_masked distribution: {torch.bincount(preds_masked).tolist() if len(preds_masked) > 0 else 'N/A'}")
            # Check accuracy
            if len(label_masked) > 0:
                acc = (preds_masked == label_masked).float().mean().item()
                print(f"[ESMC] Accuracy (on this batch): {acc:.6f}")
            print(f"[ESMC] Final loss: {loss.item():.6f}")
            # Check gradients
            if logits.requires_grad:
                print(f"[ESMC] logits requires_grad: True")
                if logits.grad is not None:
                    print(f"[ESMC] logits.grad mean: {logits.grad.mean().item():.6f}")
            else:
                print(f"[ESMC] logits requires_grad: False")
            print(f"[ESMC] ========== Loss Function End ==========\n")

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

    def on_train_epoch_end(self):
        """Print classifier weight changes at the end of each training epoch"""
        super().on_train_epoch_end()
        
        # Get the classifier head - this should return modules_to_save.default if LoRA is used
        head = self._get_head()
        
        # CRITICAL: When LoRA is used with modules_to_save, _get_head() should return
        # modules_to_save.default, but we need to verify we're checking the right weight.
        # Also check the base model structure to find the actual classifier being used.
        base_model = self._get_base_model()
        actual_classifier = None
        classifier_weight = None
        
        # Try to find the actual classifier weight that's being updated
        if hasattr(base_model, 'classifier'):
            classifier = base_model.classifier
            # If LoRA wrapped, check modules_to_save.default
            if hasattr(classifier, 'modules_to_save') and hasattr(classifier.modules_to_save, 'default'):
                actual_classifier = classifier.modules_to_save.default
                if hasattr(actual_classifier, 'weight'):
                    classifier_weight = actual_classifier.weight
            # If not wrapped or if it's a Linear directly
            elif hasattr(classifier, 'weight'):
                actual_classifier = classifier
                classifier_weight = classifier.weight
            # If it's a Sequential, find the last Linear layer
            elif hasattr(classifier, '__getitem__'):
                for i in range(len(classifier) - 1, -1, -1):
                    if hasattr(classifier[i], 'weight'):
                        actual_classifier = classifier[i]
                        classifier_weight = classifier[i].weight
                        break
        
        # Fallback to head if we couldn't find it above
        if classifier_weight is None and hasattr(head, 'weight'):
            actual_classifier = head
            classifier_weight = head.weight
        
        # Store initial weights if not already stored
        if not hasattr(self, '_initial_weight_std') and classifier_weight is not None:
            self._initial_weight_std = classifier_weight.std().item()
            self._initial_weight_mean = classifier_weight.mean().item()
            self._initial_weight = classifier_weight.data.clone()
        
        # Print current weight stats
        if classifier_weight is not None:
            current_mean = classifier_weight.mean().item()
            current_std = classifier_weight.std().item()
            epoch = getattr(self, 'epoch', 0)
            
            print(f"\n[ESMC] ========== Epoch {epoch} End - Classifier Weight Stats ==========")
            print(f"[ESMC] Classifier type: {type(actual_classifier)}")
            print(f"[ESMC] Current weight - mean: {current_mean:.6f}, std: {current_std:.6f}")
            
            if hasattr(self, '_initial_weight_std'):
                mean_change = current_mean - self._initial_weight_mean
                std_change = current_std - self._initial_weight_std
                std_change_pct = (std_change / self._initial_weight_std) * 100 if self._initial_weight_std > 0 else 0
                print(f"[ESMC] Change from initial - mean: {mean_change:+.6f}, std: {std_change:+.6f} ({std_change_pct:+.2f}%)")
                
                # Check actual weight difference (more reliable than mean/std)
                if hasattr(self, '_initial_weight'):
                    weight_diff = (classifier_weight.data - self._initial_weight).abs().mean().item()
                    weight_max_diff = (classifier_weight.data - self._initial_weight).abs().max().item()
                    print(f"[ESMC] Weight absolute difference - mean: {weight_diff:.8f}, max: {weight_max_diff:.8f}")
                
                # Check if weight has gradients (should be None after optimizer step, but param should have .grad during backward)
                if hasattr(classifier_weight, 'grad') and classifier_weight.grad is not None:
                    print(f"[ESMC] WARNING: Weight still has gradient after optimizer step (should be None)")
                elif hasattr(actual_classifier, 'weight') and actual_classifier.weight.requires_grad:
                    print(f"[ESMC] Weight requires_grad: True (should be trainable)")
            
            # Also check if there's a modules_to_save version for comparison
            if hasattr(base_model, 'classifier') and hasattr(base_model.classifier, 'modules_to_save'):
                if hasattr(base_model.classifier.modules_to_save, 'default'):
                    mtos_head = base_model.classifier.modules_to_save.default
                    if hasattr(mtos_head, 'weight'):
                        mtos_mean = mtos_head.weight.mean().item()
                        mtos_std = mtos_head.weight.std().item()
                        print(f"[ESMC] modules_to_save.default - mean: {mtos_mean:.6f}, std: {mtos_std:.6f}")
                        if actual_classifier is not mtos_head:
                            print(f"[ESMC] WARNING: Checking different classifier than modules_to_save.default!")
            
            # Verify this weight is in the optimizer
            if hasattr(self, 'optimizer') and classifier_weight is not None:
                in_optimizer = False
                optimizer_lr = None
                for group in self.optimizer.param_groups:
                    # Use 'is' to check object identity, not 'in' which causes tensor comparison
                    if any(classifier_weight is p for p in group['params']):
                        in_optimizer = True
                        optimizer_lr = group.get('lr', None)
                        break
                print(f"[ESMC] Weight in optimizer: {in_optimizer}")
                if optimizer_lr is not None:
                    print(f"[ESMC] Optimizer learning rate for this weight: {optimizer_lr}")
            
            # Check if there's a sequence_head that might be interfering
            if hasattr(base_model, 'sequence_head'):
                print(f"[ESMC] Found sequence_head in base_model: {type(base_model.sequence_head)}")
                # Check if sequence_head parameters are in optimizer
                if hasattr(self, 'optimizer'):
                    seq_head_params = list(base_model.sequence_head.parameters())
                    seq_head_in_opt_count = 0
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if any(p is sp for sp in seq_head_params):
                                seq_head_in_opt_count += 1
                    print(f"[ESMC] sequence_head parameters in optimizer: {seq_head_in_opt_count}/{len(seq_head_params)}")
            
            print(f"[ESMC] ================================================================\n")
        else:
            print(f"\n[ESMC] ========== Epoch {epoch} End - Classifier Weight Stats ==========")
            print(f"[ESMC] WARNING: Could not find classifier weight to check!")
            print(f"[ESMC] ================================================================\n")

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
        mcc = (tp * tn - fp * fn) / (
            (tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt()
        )
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