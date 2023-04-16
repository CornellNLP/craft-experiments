import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class PositionalFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1, device=torch.device('cpu')):
        super().__init__()

        self._device = device

        # Define two feed-forward (linear) layers.
        self._linear_initial = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, device=self._device)
        nn.init.xavier_normal_(self._linear_initial.weight.data)  # initialize query weights with Xavier distribution
        self._linear_final = nn.Linear(in_features=hidden_dim, out_features=embedding_dim, device=self._device)
        nn.init.xavier_normal_(self._linear_final.weight.data)  # initialize query weights with Xavier distribution

        self._layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, device=self._device)
        self._dropout = nn.Dropout(p=dropout)
        self._gelu = nn.GELU()

    @property
    def type(self):
        return 'feed_forward'

    def forward(self, inputs):
        inputs = device_mapper(inputs, self._device)

        # inputs: (batch_size, max_length, embedding_dim)
        # _linear_initial: (batch_size, max_length, hidden_dim)
        outputs = self._linear_initial(inputs)
        outputs = self._gelu(outputs)
        # outputs: (batch_size, max_length, embedding_dim)
        outputs = self._linear_final(outputs)
        outputs = self._dropout(outputs)

        residual_add = self._layer_norm(inputs + outputs)
        return residual_add
