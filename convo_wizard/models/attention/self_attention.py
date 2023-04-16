import numpy as np
import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dropout=0.05, device=torch.device('cpu')):
        super().__init__()

        self._device = device

        self._temperature = np.power(embedding_dim, 0.5)
        self._dropout = nn.Dropout(p=attention_dropout)
        self._softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _get_attention_padding_mask(query_attention_mask):
        # query_attention_mask: (batch_size, max_length)
        # note: [PAD] tokens are marked with 1 and non-[PAD] tokens with 0.
        # batch_size = query_attention_mask.shape[0]
        max_length = query_attention_mask.shape[-1]

        padding_attention_mask = query_attention_mask.unsqueeze(-2)
        # padding_attention_mask: (batch_size, 1, max_length)
        repeat_shape = [1] * len(padding_attention_mask.shape)
        repeat_shape[-2] = max_length
        padding_attention_mask = padding_attention_mask.repeat(repeat_shape)
        return padding_attention_mask.bool()

    def forward(self, key, query, value, query_attention_mask):
        key = device_mapper(key, self._device)
        query = device_mapper(query, self._device)
        value = device_mapper(value, self._device)
        query_attention_mask = device_mapper(query_attention_mask, self._device)

        # query, key, value: (batch_size, max_length, query/key/value_dim = embedding_dim)
        # attention_scores: (batch_size, max_length, max_length) = (QK') / sqrt(d)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self._temperature

        # query_attention_mask: (batch_size, max_length)
        # padding_attention_mask: (batch_size, max_length, max_length)
        padding_attention_mask = self._get_attention_padding_mask(query_attention_mask)

        # Replace all [PAD] token attention scores as -inf.
        attention_scores.masked_fill_(padding_attention_mask, float(-np.inf))
        # attention_filter: (batch_size, max_length, max_length)
        attention_filter = self._softmax(attention_scores)
        attention_filter = self._dropout(attention_filter)

        # context: (batch_size, max_length, value_dim)
        context = torch.matmul(attention_filter, value)

        return context, attention_filter
