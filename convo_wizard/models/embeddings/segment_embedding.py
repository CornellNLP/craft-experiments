import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class SegmentEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_token_types=2, pad_token_type=0, device=torch.device('cpu')):
        super().__init__()

        self._num_token_types = num_token_types + 1  # [PAD] token
        self._embedding_dim = embedding_dim
        self._pad_token_type = pad_token_type
        self._device = device

        # Initialize interval segment (token_type) embeddings, and initialize weights using the
        # Xavier distribution.
        self._segment_embedding = nn.Embedding(num_embeddings=self._num_token_types, embedding_dim=embedding_dim,
                                               padding_idx=pad_token_type, device=device)
        nn.init.xavier_normal_(self._segment_embedding.weight.data)  # note: resets padding_idx weights
        self._segment_embedding.weight.data[pad_token_type] = torch.zeros(embedding_dim)

    @property
    def type(self):
        return 'segment_embedding'

    def resize_segment_embedding(self, num_new_token_types=0):
        _resized_segment_embedding = nn.Embedding(num_embeddings=(self._num_token_types + num_new_token_types),
                                                  embedding_dim=self._embedding_dim, padding_idx=self._pad_token_type,
                                                  device=self._device)
        _resized_segment_embedding.weight.data[:self._num_token_types] = self._segment_embedding.weight.data
        self._segment_embedding = _resized_segment_embedding
        self._num_token_types = self._num_token_types + num_new_token_types

    def forward(self, token_type_ids):
        # token_type_ids: (batch_size, max_length)
        token_type_ids = device_mapper(token_type_ids, self._device)

        # segment_embeddings: (batch_size, max_length, embedding_dim)
        segment_embeddings = self._segment_embedding(token_type_ids)

        return segment_embeddings
