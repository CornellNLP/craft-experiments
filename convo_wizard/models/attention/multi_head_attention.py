import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper
from self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, attention_dropout=0.05, dropout=0.1,
                 device=torch.device('cpu')):
        super().__init__()

        self._device = device
        self._num_heads = num_heads

        assert embedding_dim % self._num_heads == 0
        self._key_dim = embedding_dim // self._num_heads
        self._query_dim = embedding_dim // self._num_heads
        self._value_dim = embedding_dim // self._num_heads

        self._key_projection = nn.Linear(in_features=embedding_dim, out_features=(self._num_heads * self._key_dim),
                                         device=self._device)
        nn.init.xavier_normal_(self._key_projection.weight.data)  # initialize key weights with Xavier distribution
        self._query_projection = nn.Linear(in_features=embedding_dim, out_features=(self._num_heads * self._query_dim),
                                           device=self._device)
        nn.init.xavier_normal_(self._query_projection.weight.data)  # initialize query weights with Xavier distribution
        self._value_projection = nn.Linear(in_features=embedding_dim, out_features=(self._num_heads * self._value_dim),
                                           device=self._device)
        nn.init.xavier_normal_(self._value_projection.weight.data)  # initialize value weights with Xavier distribution

        # Initialize the self-attention module and final linear projection.
        self._self_attention = SelfAttention(embedding_dim=embedding_dim, attention_dropout=attention_dropout,
                                             device=self._device)
        self._attention_filters = None
        self._projection = nn.Linear(self._num_heads * self._value_dim, embedding_dim, device=self._device)

        # Normalize the weights and add dropout.
        self._layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12, device=self._device)
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, query_attention_mask):
        key = device_mapper(key, self._device)
        query = device_mapper(query, self._device)
        value = device_mapper(value, self._device)
        query_attention_mask = device_mapper(query_attention_mask, self._device)

        # query, key, value: (batch_size, max_length, query/key/value_dim = embedding_dim)
        batch_size = key.shape[0]
        max_length = key.shape[1]

        # projection: (batch_size, max_length, num_heads * key/query/value_dim)
        # .view: (batch_size, max_length, num_heads, key/query/value_dim)
        # .permute: (batch_size, num_heads, max_length, key/query/value_dim)
        key_projection = self._key_projection(key) \
            .view(batch_size, max_length, self._num_heads, self._key_dim) \
            .permute(0, 2, 1, 3)
        query_projection = self._query_projection(query) \
            .view(batch_size, max_length, self._num_heads, self._query_dim) \
            .permute(0, 2, 1, 3)
        value_projection = self._query_projection(value) \
            .view(batch_size, max_length, self._num_heads, self._value_dim) \
            .permute(0, 2, 1, 3)

        # query_attention_mask: (batch_size, max_length)
        # padding_attention_mask: (batch_size, num_heads, max_length)
        padding_attention_mask = query_attention_mask.unsqueeze(1).expand(batch_size, self._num_heads, max_length)
        # contexts: (batch_size, num_heads, max_length, value_dim)
        # attention_filters: (batch_size, num_heads, max_length, max_length)
        contexts, attention_filters = \
            self._self_attention(key=key_projection, query=query_projection, value=value_projection,
                                 query_attention_mask=padding_attention_mask)

        # .permute: (batch_size, max_length, num_heads, value_dim)
        # .reshape: (batch_size, max_length, num_heads * value_dim)
        contexts = contexts.permute(0, 2, 1, 3).reshape(batch_size, max_length, -1)
        # outputs: (batch_size, max_length, embedding_dim)
        outputs = self._projection(contexts)
        outputs = self._dropout(outputs)

        # Add residual connection to overcome any vanishing gradients.
        residual_add = self._layer_norm(query + outputs)
        self._attention_filters = attention_filters

        return residual_add, attention_filters
