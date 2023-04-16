import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class LinearLanguageModelingHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size, device=torch.device('cpu')):
        super().__init__()

        self._device = device

        self._linear_lm_head = nn.Linear(in_features=embedding_dim, out_features=vocab_size, device=self._device)
        nn.init.xavier_normal_(self._linear_lm_head.weight)
        self._linear_lm_head.bias.data.fill_(0.0)

    @property
    def type(self):
        return 'linear_language_model'

    def forward(self, inputs):
        inputs = device_mapper(inputs, self._device)

        # inputs: (batch_size, max_length, embedding_dim)
        # outputs: (batch_size, max_length, vocab_size)
        outputs = self._linear_lm_head(inputs)
        return outputs
