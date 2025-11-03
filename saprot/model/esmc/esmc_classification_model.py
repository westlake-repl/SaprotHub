import torchmetrics
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from ..model_interface import register_model
from .base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

        # 确保模型有一个分类器头
        # 注意：这里的 self.model 是在 ESMCBaseModel 中加载的预训练模型
        # 我们需要为它添加或替换一个适合我们任务的分类头
        # 假设基础模型有一个名为 `embed_dim` 的属性
        embed_dim = self.model.embed_dim
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
            # ======================= 核心修正流程 =======================

            # === 修正点 1: 正确的 Tokenization 方式 ===
            # 之前错误的方式: [self.model.embed(p) for p in proteins]
            # 正确的方式: 从 ESMProtein 对象中获取序列字符串 (p.sequence)，然后使用 tokenizer
            token_ids_list = [self.model.tokenizer(p.sequence) for p in proteins]

            # 步骤 2: 填充 (Padding)
            token_ids_batch = pad_sequence(
                token_ids_list,
                batch_first=True,
                padding_value=self.model.padding_idx
            )
            
            # 将张量移动到正确的设备 (非常重要！)
            token_ids_batch = token_ids_batch.to(self.device)

            # 步骤 3: 模型推理 (Inference)
            model_output = self.model.forward(
                token_ids_batch,
                repr_layers=[self.model.num_layers]
            )
            representations = model_output['representations'][self.model.num_layers]
            
            # 步骤 4: 池化 (Pooling) 以获得单个序列表示
            mask = (token_ids_batch != self.model.padding_idx).unsqueeze(-1)
            sequence_lengths = mask.sum(dim=1)
            # 增加健壮性：防止因序列长度为0而导致除以零的错误
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths
            # ================================================================

        # === 修正点 2: 更稳健的分类器调用方式 ===
        # 之前脆弱的方式:
        # x = self.model.classifier[0](pooled_repr)
        # ...
        # 正确的方式: 直接调用整个分类器模块
        logits = self.model.classifier(pooled_repr)

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        # 获取当前阶段的 metrics 字典
        stage_metrics = self.metrics[stage]
        for metric in stage_metrics.values():
            metric.update(logits.detach(), label)

        if stage == "train":
            # 在训练步骤中，我们只记录损失，指标在验证步骤中记录更有意义
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # 在验证周期结束时计算并记录指标
        log_dict = self.get_log_dict("valid")
        self.log_info(log_dict)
        self.reset_metrics("valid")

    def get_log_dict(self, stage):
        log_dict = {}
        stage_metrics = self.metrics.get(stage, {})
        for name, metric in stage_metrics.items():
            # .compute() 获取指标的最终值
            log_dict[name] = metric.compute()
        return log_dict

    def reset_metrics(self, stage):
        stage_metrics = self.metrics.get(stage, {})
        for metric in stage_metrics.values():
            # .reset() 清空累计的指标状态
            metric.reset()



