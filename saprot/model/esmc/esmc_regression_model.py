import torchmetrics
import torch.distributed as dist
import torch

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
        # =================================================================
        # START: DETAILED DEBUGGING PRINTS
        # =================================================================
        print("\n" + "="*40)
        print("--- ENTERING ESMCRegressionModel FORWARD PASS ---")
        print(f"Step 1: Inspecting self.model object")
        print(f"  - Type of self.model: {type(self.model)}")
        # 在 LoRA 场景下，我们期望看到 <class 'peft.peft_model.PeftModel'>
        
        # Parse proteins input
        proteins = self._parse_proteins_input(inputs)

        # Tokenization & Padding
        token_ids_batch, attention_mask, tokenizer = self._tokenize_sequences(proteins)

        # Backbone representations
        representations = self._get_representations(token_ids_batch)

        # Pooling
        pooled_repr = self._pool_representations(representations, token_ids_batch, tokenizer.pad_token_id)

        print("\nStep 2: Acquiring the head module")
        base_model = self._get_base_model()
        print(f"  - Type of base_model (from self._get_base_model()): {type(base_model)}")
        # 我们期望看到 <class 'esm.sdk.api.ESMProtein'> 或类似的底层模型

        head = None
        
        # --- Path A: Manual Search Logic ---
        print("\nStep 3: Executing the manual search logic (if/elif block)")
        
        # Check for 'classifier'
        if hasattr(base_model, 'classifier'):
            print("  - Found 'classifier' attribute on base_model.")
            print(f"    - Type of base_model.classifier: {type(base_model.classifier)}")
            # 在 LoRA 场景下，我们期望看到 <class 'peft.tuners.lora.Linear'>
            if hasattr(base_model.classifier, 'modules_to_save'):
                print("    - 'classifier' has 'modules_to_save' attribute.")
                if hasattr(base_model.classifier.modules_to_save, 'default'):
                    print("    - 'modules_to_save' has 'default' attribute. Assigning it to head.")
                    head = base_model.classifier.modules_to_save.default
                    print(f"    - Head found via manual search (classifier path): {type(head)}")
                else:
                    print("    - 'modules_to_save' does NOT have 'default' attribute.")
            else:
                print("    - 'classifier' does NOT have 'modules_to_save' attribute.")
        else:
            print("  - Did NOT find 'classifier' attribute on base_model.")

        # Check for 'head' if classifier path failed
        if head is None and hasattr(base_model, 'head'):
            print("  - Found 'head' attribute on base_model.")
            print(f"    - Type of base_model.head: {type(base_model.head)}")
            if hasattr(base_model.head, 'modules_to_save'):
                print("    - 'head' has 'modules_to_save' attribute.")
                if hasattr(base_model.head.modules_to_save, 'default'):
                    print("    - 'modules_to_save' has 'default' attribute. Assigning it to head.")
                    head = base_model.head.modules_to_save.default
                    print(f"    - Head found via manual search (head path): {type(head)}")
                else:
                    print("    - 'modules_to_save' does NOT have 'default' attribute.")
            else:
                print("    - 'head' does NOT have 'modules_to_save' attribute.")
        elif head is None:
            print("  - Did NOT find 'head' attribute on base_model.")

        print(f"\nStep 4: Result of manual search")
        print(f"  - 'head' object after manual search: {type(head)}")

        # --- Path B: The self._get_head() helper function ---
        print("\nStep 5: Calling the helper function self._get_head() for comparison")
        head_from_helper = self._get_head()
        print(f"  - 'head' object returned by self._get_head(): {type(head_from_helper)}")

        # --- Final Decision Logic ---
        if head is None:
            print("\nStep 6: Manual search failed. Falling back to self._get_head().")
            head = self._get_head()
        else:
            print("\nStep 6: Manual search succeeded. Using the manually found head.")

        print(f"\nStep 7: Final 'head' object being used for computation")
        print(f"  - Final head type: {type(head)}")
        # 我们期望看到 <class 'torch.nn.modules.linear.Linear'>

        # =================================================================
        # END: DETAILED DEBUGGING PRINTS
        # =================================================================
        
        logits = head(pooled_repr).squeeze(dim=-1)
        
        print("--- FORWARD PASS COMPLETED ---")
        print("="*40 + "\n")

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