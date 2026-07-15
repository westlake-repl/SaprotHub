import torch

from .base import ProSSTBaseModel


class ProSSTPairBaseModel(ProSSTBaseModel):
    def __init__(self, task: str, output_size: int, **kwargs):
        self.pair_output_size = output_size
        super().__init__(task=task, **kwargs)

    def initialize_model(self):
        super().initialize_model()
        pair_hidden_size = self.model.config.hidden_size * 2
        classifier = torch.nn.Sequential(
            torch.nn.Linear(pair_hidden_size, pair_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(pair_hidden_size, self.pair_output_size),
        )
        setattr(self.model, "classifier", classifier)

    def forward(self, inputs_1, inputs_2):
        representation_1 = self.get_pooled_representations(inputs_1)
        representation_2 = self.get_pooled_representations(inputs_2)
        pair_representation = torch.cat(
            [representation_1, representation_2],
            dim=-1,
        )
        return self._get_classifier()(pair_representation)
