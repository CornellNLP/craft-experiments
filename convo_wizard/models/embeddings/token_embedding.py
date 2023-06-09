import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0, device=torch.device('cpu')):
        super().__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx
        self._device = device

        # Initialize token embeddings and weights using the Xavier distribution.
        self._token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0,
                                             device=self._device)
        nn.init.xavier_uniform_(self._token_embedding.weight.data)  # note: resets padding_idx weights
        self._token_embedding.weight.data[padding_idx] = torch.zeros(embedding_dim)

    @property
    def type(self):
        return 'token_embedding'

    @property
    def embedding(self):
        return self._token_embedding

    def resize_token_embedding(self, num_new_tokens=0):
        _resized_token_embedding = nn.Embedding(num_embeddings=(self._vocab_size + num_new_tokens),
                                                embedding_dim=self._embedding_dim, padding_idx=self._padding_idx,
                                                device=self._device)
        _resized_token_embedding.weight.data[:self._vocab_size] = self._token_embedding.weight.data
        self._token_embedding = _resized_token_embedding
        self._vocab_size = self._vocab_size + num_new_tokens

    def forward(self, input_ids):
        # input_ids: (batch_size, max_length)
        input_ids = device_mapper(input_ids, self._device)

        # token_embeddings: (batch_size, max_length, embedding_dim)
        token_embeddings = self._token_embedding(input_ids)

        return token_embeddings
