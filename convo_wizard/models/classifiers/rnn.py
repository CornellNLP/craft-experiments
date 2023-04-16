import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class RecurrentClassifierHead(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_dim=1024, rnn_bidirectional=True, output_dim=2,
                 device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device

        self._num_directions = 2 if rnn_bidirectional else 1
        self._hidden_dim = rnn_hidden_dim
        self._lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self._hidden_dim, bidirectional=rnn_bidirectional,
                             batch_first=True, bias=True, device=self._device)

        self._classifier = nn.Linear(in_features=self._num_directions * self._hidden_dim, out_features=output_dim,
                                     device=self._device)
        nn.init.xavier_normal_(self._classifier.weight)
        self._classifier.bias.data.fill_(0.0)

    @property
    def type(self):
        return 'recurrent_classifier'

    def forward(self, inputs):
        inputs = device_mapper(inputs, self._device)

        # inputs: (batch_size, max_length, embedding_dim)
        # lstm_outputs: (batch_size, max_length, _num_directions * hidden_dim)
        # outputs: (batch_size, max_len, output_dim)
        lstm_outputs, _ = self._lstm(inputs)
        outputs = self._classifier(lstm_outputs)

        return outputs
