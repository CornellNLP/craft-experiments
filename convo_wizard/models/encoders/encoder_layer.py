import torch
from torch import nn

from convo_wizard.models.attention.multi_head_attention import MultiHeadAttention
from convo_wizard.models.positional_networks.convolution import PositionalConvolution
from convo_wizard.models.positional_networks.feed_forward import PositionalFeedForward
from convo_wizard.utils.utils import device_mapper


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, hidden_dim, positional_network_type='conv', attention_dropout=0.05,
                 dropout=0.1, device=torch.device('cpu')):
        super().__init__()

        self._device = device

        self._multi_head_attention = \
            MultiHeadAttention(num_heads=num_heads, embedding_dim=embedding_dim, attention_dropout=attention_dropout,
                               dropout=dropout, device=self._device)
        if positional_network_type == 'conv':
            self._positional_network = PositionalConvolution(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                                             dropout=dropout, device=self._device)
        else:
            self._positional_network = PositionalFeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                                             dropout=dropout, device=self._device)

    def forward(self, embeddings, attention_mask):
        embeddings = device_mapper(embeddings, self._device)
        attention_mask = device_mapper(attention_mask, self._device)

        # embeddings: (batch_size, max_length, embedding_dim)
        # attention_mask: (batch_size, max_length)
        # multi_head_attention_output, outputs: (batch_size, max_length, embedding_dim)
        multi_head_attention_output, attention_filters = \
            self._multi_head_attention(query=embeddings, key=embeddings, value=embeddings,
                                       query_attention_mask=attention_mask)
        outputs = self._positional_network(inputs=multi_head_attention_output)

        return outputs, attention_filters
