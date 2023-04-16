import torch
from torch import nn

from convo_wizard.models.embeddings.positional_encoding import PositionalEncoding
from convo_wizard.utils.utils import device_mapper


class RelativePositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_relative_position=100, use_sinusoidal_init=True, pad_token_position=0,
                 freq_base=5000, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device
        self._max_relative_position = max_relative_position
        num_positions = self._max_relative_position + 2  # one for [PAD] token and one for [UNK]

        # The term 'embedding' implies that it is trainable, while the term 'encoding' implies that they are static.
        self._position_embedding = nn.Embedding(num_embeddings=num_positions, embedding_dim=embedding_dim,
                                                padding_idx=pad_token_position, device=self._device)
        if use_sinusoidal_init:
            PositionalEncoding.sinusoidal_weight_init(num_positions=num_positions, embedding_dim=embedding_dim,
                                                      freq_base=freq_base, pad_token_position=pad_token_position,
                                                      device=self._device)
        else:
            nn.init.xavier_normal_(self._position_embedding.weight.data)  # note: resets padding_idx weights
            self._position_embedding.weight.data[pad_token_position] = torch.zeros(embedding_dim)
        self._position_embedding.weight.requires_grad = True  # do NOT freeze these relative embeddings

    @property
    def type(self):
        return 'relative_positional_embedding'

    def _transform_relative_position_ids(self, relative_position_ids):
        # relative_position_ids: (batch_size, max_length)
        relative_position_ids = torch.where(relative_position_ids > self._max_relative_position,
                                            self._max_relative_position, relative_position_ids)
        return relative_position_ids

    def forward(self, relative_position_ids):
        # relative_position_ids: (batch_size, max_length)
        relative_position_ids = device_mapper(relative_position_ids, self._device)
        relative_position_ids = self._transform_relative_position_ids(relative_position_ids)

        # position_embeddings: (batch_size, max_length, embedding_dim)
        position_embeddings = self._position_embedding(relative_position_ids)
        assert position_embeddings.requires_grad is True

        return position_embeddings
