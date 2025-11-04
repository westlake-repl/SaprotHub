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
        if isinstance(inputs, dict) and 'proteins' in inputs:
            proteins = inputs['proteins']
        else:
            raise ValueError("ESMCClassificationModel.forward expects inputs['proteins'] (list of ESMProtein)")

        if not isinstance(proteins, list):
            proteins = [proteins]

        with (torch.no_grad() if self.freeze_backbone else torch.enable_grad()):
            token_ids_list = [self.model.tokenizer(p.sequence) for p in proteins]

            # 步骤 2: 填充 (Padding)
            token_ids_batch = pad_sequence(
                token_ids_list,
                batch_first=True,
                padding_value=self.model.tokenizer.alphabet.padding_idx
            )
            
            token_ids_batch = token_ids_batch.to(self.device)

            # 步骤 3: 模型推理 (Inference)
            model_output = self.model.forward(
                token_ids_batch,
                repr_layers=[len(self.model.transformer.layers)]
            )
            representations = model_output['representations'][len(self.model.transformer.layers)]
            
            # 步骤 4: 池化 (Pooling)
            mask = (token_ids_batch != self.model.tokenizer.alphabet.padding_idx).unsqueeze(-1)
            
            sequence_lengths = mask.sum(dim=1)
            sequence_lengths = sequence_lengths.clamp(min=1)
            
            pooled_repr = (representations * mask).sum(dim=1) / sequence_lengths

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