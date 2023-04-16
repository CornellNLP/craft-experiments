import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class LinearClassifierHead(nn.Module):
    def __init__(self, embedding_dim, output_dim=2, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device

        self._classifier = nn.Linear(in_features=embedding_dim, out_features=output_dim, device=self._device)
        nn.init.xavier_normal_(self._classifier.weight)
        self._classifier.bias.data.fill_(0.0)

    @property
    def type(self):
        return 'linear_classifier'

    def forward(self, inputs):
        inputs = device_mapper(inputs, self._device)

        # inputs: (batch_size, max_length, embedding_dim)
        # outputs: (batch_size, max_length, output_dim)
        outputs = self._classifier(inputs)
        return outputs
