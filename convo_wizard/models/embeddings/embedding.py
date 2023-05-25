import torch
from torch import nn

from convo_wizard.models.embeddings.positional_encoding import PositionalEncoding
from convo_wizard.models.embeddings.relative_positional_embedding import RelativePositionEmbedding
from convo_wizard.models.embeddings.segment_embedding import SegmentEmbedding
from convo_wizard.models.embeddings.token_embedding import TokenEmbedding
from convo_wizard.utils.utils import device_mapper


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0, cls_or_sep_token_idx=2, max_length=2048,
                 max_relative_position=None, pad_token_position=0, pad_token_type=0, num_token_types=2, freq_base=10000,
                 dropout=0.1, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device
        self._cls_or_sep_token_idx = cls_or_sep_token_idx  # for attention plots
        self._max_relative_position = max_relative_position

        self._token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                               padding_idx=padding_idx, device=self._device)
        if max_relative_position is None:
            self._position_encoding_or_embedding = PositionalEncoding(embedding_dim=embedding_dim,
                                                                      pad_token_position=pad_token_position,
                                                                      max_length=max_length, freq_base=freq_base,
                                                                      device=self._device)
        else:
            self._position_encoding_or_embedding = RelativePositionEmbedding(embedding_dim=embedding_dim,
                                                                             max_relative_position=self._max_relative_position,
                                                                             pad_token_position=pad_token_position,
                                                                             freq_base=freq_base, device=self._device,
                                                                             **kwargs)
        self._segment_embedding = SegmentEmbedding(embedding_dim=embedding_dim, num_token_types=num_token_types,
                                                   pad_token_type=pad_token_type, device=self._device)

        # Normalize the weights on the embedding and add dropout.
        self._layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12, device=self._device)
        self._dropout = nn.Dropout(p=dropout)

    @property
    def type(self):
        return 'embedding'

    @property
    def token_embedding(self):
        return self._token_embedding.embedding

    def forward(self, input_ids, position_ids, token_type_ids):
        input_ids = device_mapper(input_ids, self._device)
        position_ids = device_mapper(position_ids, self._device)
        token_type_ids = device_mapper(token_type_ids, self._device)

        # input_ids, position_ids, token_type_ids: (batch_size, max_length)
        token_embeddings = self._token_embedding(input_ids.long())
        position_encodings_or_embeddings = self._position_encoding_or_embedding(position_ids.long())
        segment_embeddings = self._segment_embedding(token_type_ids.long())

        embeddings = token_embeddings + position_encodings_or_embeddings + segment_embeddings
        embeddings = self._layer_norm(embeddings)
        embeddings = self._dropout(embeddings)

        # embeddings: (batch_size, max_length, embedding_dim)
        return embeddings
