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

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

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
            # 目标：将 `proteins` (一个Python列表) 转换成一个单一的、符合模型输入要求的 Tensor。

            # 步骤 1: 分词 (Tokenization)
            # 遍历列表中的每一个蛋白质对象 `p`。
            # 调用 `self.model.embed(p)` 会将蛋白质序列转换成一个代表氨基酸索引的 LongTensor。
            # `token_ids_list` 现在是一个包含了多个 Tensor 的列表，例如 [Tensor([3, 1, 4]), Tensor([5, 9, 2, 6])]
            token_ids_list = [self.model.embed(p) for p in proteins]

            # 步骤 2: 填充 (Padding)
            # 由于列表中的 Tensor 长度不同（因为蛋白质序列长度不同），不能直接堆叠。
            # `pad_sequence` 函数会将这个 Tensor 列表打包成一个规整的批次张量。
            # `batch_first=True` 表示输出张量的形状是 [批次大小, 最长序列长度]。
            # `padding_value` 指定用什么值来填充较短的序列，这里使用模型预设的填充索引。
            token_ids_batch = pad_sequence(
                token_ids_list,
                batch_first=True,
                padding_value=self.model.padding_idx
            )
            # `token_ids_batch` 现在是一个单一的 LongTensor，例如 Tensor([[3, 1, 4, 0], [5, 9, 2, 6]])
            # 这正是 `embedding()` 函数所期望的 `indices` 参数类型！

            # 步骤 3: 模型推理 (Inference)
            # 现在，我们可以将这个格式正确的批次张量 `token_ids_batch` 传递给模型。
            # 这次调用将成功执行，不会再产生 TypeError。
            model_output = self.model.forward(
                token_ids_batch,
                repr_layers=[self.model.num_layers]
            )
            representations = model_output['representations'][self.model.num_layers]
            
            # 步骤 4: 池化 (Pooling) 以获得单个序列表示
            # 为了进行分类，我们需要将每个序列（长度可变）的输出表示（形状为 [批次大小, 序列长度, 嵌入维度]）
            # 转换成一个固定大小的向量（形状为 [批次大小, 嵌入维度]）。
            # 这里我们使用平均池化，但需要忽略填充部分。
            mask = (token_ids_batch != self.model.padding_idx).unsqueeze(-1)
            sequence_lengths = mask.sum(dim=1)
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths
            # ================================================================

        # 将池化后的表示送入分类头
        x = self.model.classifier[0](pooled_repr)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        logits = self.model.classifier[3](x)

        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")

        return loss



