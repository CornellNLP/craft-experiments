import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class PositionalConvolution(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1, device=torch.device('cpu')):
        super().__init__()

        self._device = device

        # Define two convolutional layers with kernel size of 1.
        # Creates an "hour-glass" network when hidden_dim < embedding_dim
        self._conv_initial = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=1,
                                       device=self._device)
        self._conv_final = nn.Conv1d(in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=1,
                                     device=self._device)

        self._layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, device=self._device)
        self._dropout = nn.Dropout(p=dropout)
        self._gelu = nn.GELU()

    @property
    def type(self):
        return 'convolution'

    def forward(self, inputs):
        inputs = device_mapper(inputs, self._device)

        # inputs: (batch_size, max_length, embedding_dim)
        # .permute: (batch_size, embedding_dim, max_length)
        # _conv_initial: (batch_size, hidden_dim, max_length)
        outputs = self._conv_initial(inputs.permute(0, 2, 1))
        outputs = self._gelu(outputs)
        # _conv_final: (batch_size, embedding_dim, max_length)
        # .permute: (batch_size, max_length, embedding_dim)
        outputs = self._conv_final(outputs).permute(0, 2, 1)
        # outputs: (batch_size, max_length, embedding_dim)
        outputs = self._dropout(outputs)

        residual_add = self._layer_norm(inputs + outputs)
        return residual_add
