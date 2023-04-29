import torch
from torch import nn

from convo_wizard.models.embeddings.embedding import Embedding
from convo_wizard.models.encoders.encoder_layer import EncoderLayer
from convo_wizard.utils.utils import device_mapper


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_relative_position=None,
                 positional_network_type='conv', padding_idx=0, cls_token_idx=2, num_heads=3, num_encoder_layers=6,
                 max_length=2048, pad_token_position=0, pad_tok_type=0, num_token_types=2, attention_dropout=0.05,
                 dropout=0.1, freq_base=10000, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device
        self._max_length = max_length
        self._num_encoder_layers = num_encoder_layers

        self._embedding = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx,
                                    cls_token_idx=cls_token_idx, max_length=max_length,
                                    max_relative_position=max_relative_position, pad_token_position=pad_token_position,
                                    pad_tok_type_id=pad_tok_type, num_token_types=num_token_types,
                                    freq_base=freq_base, dropout=dropout, device=self._device, **kwargs)

        # BERT: num_heads=8, num_encoder_layers=12.
        _encoder_layer_list = []
        for _ in range(num_encoder_layers):
            _encoder_layer = EncoderLayer(num_heads=num_heads, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                          positional_network_type=positional_network_type,
                                          attention_dropout=attention_dropout, dropout=dropout, device=self._device)
            _encoder_layer_list.append(_encoder_layer)
        self._encoding_layers = nn.ModuleList(_encoder_layer_list)

    @property
    def token_embedding(self):
        return self._embedding.token_embedding

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask):
        input_ids = device_mapper(input_ids, self._device)
        position_ids = device_mapper(position_ids, self._device)
        token_type_ids = device_mapper(token_type_ids, self._device)
        attention_mask = device_mapper(attention_mask, self._device)

        # outputs: (batch_size, max_length, embedding_dim)
        outputs = self._embedding(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        outputs_all_layers, attention_filters_all_layers = [], []
        for _encoding_layer in self._encoding_layers:
            # outputs: (batch_size, max_length, embedding_dim)
            # attention_filters: (batch_size, num_heads, max_length, max_length)
            outputs, attention_filters = _encoding_layer(embeddings=outputs, attention_mask=attention_mask)
            outputs_all_layers.append(outputs)
            attention_filters_all_layers.append(attention_filters)

        return outputs_all_layers, attention_filters_all_layers
