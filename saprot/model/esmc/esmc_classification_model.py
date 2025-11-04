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

        # === 核心修正点 ===
        # 根据诊断报告 (dir(self.model) 的输出)，我们发现 self.model 有一个 'embed' 属性，
        # 它是一个 torch.nn.Embedding 层。我们可以从这个层直接获取嵌入维度。
        # 这是针对 <class 'esm.models.esmc.ESMC'> 对象的正确方法。
        embed_dim = self.model.embed.embedding_dim
        
        # 使用正确获取到的 embed_dim 来安全地创建或替换分类器。
        # 注意：我们在这里是替换模型上可能已存在的 'classifier' 属性。
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, self.num_labels)
        )

    def initialize_metrics(self, stage):
        # 使用新的 torchmetrics API，需要指定 task
        return {f"{stage}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)}

    def forward(self, inputs, coords=None):
        # 从输入字典中获取蛋白质数据列表
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            # 步骤 1: Tokenization
            # 注意：这里的 self.model.tokenizer 是在 ESMCBaseModel 中被正确设置的
            token_ids_list = [self.model.tokenizer(p.sequence) for p in proteins]

            # 步骤 2: 填充 (Padding)
            token_ids_batch = pad_sequence(
                token_ids_list,
                batch_first=True,
                padding_value=self.model.tokenizer.padding_idx
            )
            
            # 将张量移动到正确的设备
            token_ids_batch = token_ids_batch.to(self.device)

            # 步骤 3: 模型推理 (Inference)
            # 注意：底层的 esm.models.esmc.ESMC 模型的 forward 签名可能不同
            # 它直接接收 token_ids，而不是一个复杂的字典
            model_output = self.model.forward(
                token_ids_batch,
                repr_layers=[len(self.model.transformer.layers)] # 获取正确的层数
            )
            representations = model_output['representations'][len(self.model.transformer.layers)]
            
            # 步骤 4: 池化 (Pooling)
            mask = (token_ids_batch != self.model.tokenizer.padding_idx).unsqueeze(-1)
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths

        # 将池化后的表示送入我们定义的分类头
        logits = self.model.classifier(pooled_repr)

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        stage_metrics = self.metrics[stage]
        for metric in stage_metrics.values():
            metric.update(logits.detach(), label)

        if stage == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        self.log_info(log_dict)
        self.reset_metrics("valid")

    def get_log_dict(self, stage):
        log_dict = {}
        stage_metrics = self.metrics.get(stage, {})
        for name, metric in stage_metrics.items():
            log_dict[name] = metric.compute()
        return log_dict

    def reset_metrics(self, stage):
        stage_metrics = self.metrics.get(stage, {})
        for metric in stage_metrics.values():
            metric.reset()