import torchmetrics
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence

# 确保下面的导入路径与您的项目结构相符
from saprot.model.model_interface import register_model
from saprot.model.esmc.base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

        embed_dim = self.model.embed.embedding_dim
        
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, self.num_labels)
        )

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)}

    def forward(self, inputs, coords=None):
        # ====================================================================
        # ======================= 新增诊断代码注入处 =========================
        # ====================================================================
        print("\n" + "="*60)
        print("=== RUNNING DIAGNOSTIC FOR `self.model.transformer` OBJECT ===")
        print("="*60)
        try:
            transformer_obj = self.model.transformer
            print(f"\n[INFO] Inspecting object: self.model.transformer (type: {type(transformer_obj)})")
            
            # 打印 transformer 对象的属性
            transformer_attrs = [attr for attr in dir(transformer_obj) if not attr.startswith('_')]
            print(f"       Attributes of self.model.transformer: {transformer_attrs}\n")

            # 测试 transformer 对象是否支持 len()
            print("--- Testing `len()` support for `self.model.transformer` ---")
            try:
                num_layers_test = len(transformer_obj)
                print(f"   [SUCCESS] `len(self.model.transformer)` is VALID. Value: {num_layers_test}.")
                print("              This suggests the correct way to get the number of layers is `len(self.model.transformer)`.")
            except TypeError as e:
                print(f"   [FAILED]  `self.model.transformer` does not support `len()`. Error: {e}")

        except AttributeError:
            print("\n[ERROR] `self.model.transformer` does not exist! This is unexpected.")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred during the transformer diagnostic: {e}")

        print("\n" + "="*60)
        print("=== TRANSFORMER DIAGNOSTIC COMPLETE. Check output above. ===")
        print("=== The program will now continue and likely crash.      ===")
        print("="*60 + "\n")
        # ====================================================================
        # ======================= 诊断代码结束 ===========================
        # ====================================================================
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # 步骤 1 & 2: Tokenization 和 Padding (合并)
            sequences = [p.sequence for p in proteins]
            batch_encoding = self.model.tokenizer(
                sequences, 
                padding=True, 
                return_tensors="pt"
            )
            token_ids_batch = batch_encoding['input_ids'].to(self.device)

            attention_mask = batch_encoding['attention_mask'].to(self.device)

            # 步骤 3: 模型推理 (Inference)
            model_output = self.model.forward(token_ids_batch)
            representations = model_output['representations']

            # 步骤 4: 池化 (Pooling)
            mask = (token_ids_batch != self.model.tokenizer.pad_token_id).unsqueeze(-1)
            
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths

        logits = self.model.classifier(pooled_repr)

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

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

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)