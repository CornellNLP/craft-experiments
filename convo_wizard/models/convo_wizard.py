import torch
from prettytable import PrettyTable
from torch import nn

from convo_wizard.models.classifiers.linear import LinearClassifierHead
from convo_wizard.models.classifiers.rnn import RecurrentClassifierHead
from convo_wizard.models.encoders.encoder import Encoder
from convo_wizard.models.generators.linear import LinearLanguageModelingHead
from convo_wizard.utils.utils import device_mapper


class ConvoWizard(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2, max_relative_position=None, num_heads=3,
                 num_encoder_layers=6, positional_network_type='conv', classifier_head_type='rnn', padding_idx=0,
                 cls_token_idx=2, max_length=2048, pad_token_position=0, pad_token_type=0, num_token_types=2,
                 attention_dropout=0.05, dropout=0.1, freq_base=10000, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self._device = device

        self._encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                max_relative_position=max_relative_position, num_heads=num_heads,
                                num_encoder_layers=num_encoder_layers, positional_network_type=positional_network_type,
                                padding_idx=padding_idx, cls_token_idx=cls_token_idx, max_length=max_length,
                                pad_token_position=pad_token_position, pad_token_type=pad_token_type,
                                num_token_types=num_token_types, attention_dropout=attention_dropout, dropout=dropout,
                                freq_base=freq_base, device=self._device, **kwargs)

        self._lm_head = LinearLanguageModelingHead(embedding_dim=embedding_dim, vocab_size=vocab_size,
                                                   device=self._device)

        if classifier_head_type == 'rnn':
            self._classifier_head = RecurrentClassifierHead(embedding_dim=embedding_dim, output_dim=output_dim,
                                                            device=self._device, **kwargs)
        else:
            self._classifier_head = \
                LinearClassifierHead(embedding_dim=embedding_dim, output_dim=output_dim, device=self._device)

    def get_trainable_params(self):
        return (param for param in self.parameters() if param.requires_grad)

    def print_params(self, print_vals=True):
        params_table = PrettyTable(['module', 'num_params', 'requires_grad'])
        total_trainable_params = 0
        for name, param in self.named_parameters():
            params_table.add_row([name, param.numel(), param.requires_grad])
            if param.requires_grad:
                total_trainable_params = total_trainable_params + param.numel()

        if print_vals:
            print(params_table)
            print(f'total trainable params: {(total_trainable_params / 1e6):0.2f}M')
        return params_table, total_trainable_params

    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask, make_predictions=False):
        input_ids = device_mapper(input_ids, self._device)
        position_ids = device_mapper(position_ids, self._device)
        token_type_ids = device_mapper(token_type_ids, self._device)
        attention_mask = device_mapper(attention_mask, self._device)

        outputs_all_layers, attention_filters_all_layers = \
            self._encoder(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                          attention_mask=attention_mask)
        # last_layer_output: (batch_size, max_length, embedding_dim)
        last_layer_output = outputs_all_layers[-1]

        # lm_output = (batch_size, max_length, vocab_size)
        lm_output = self._lm_head(last_layer_output)

        classifier_output = None
        if make_predictions:
            # classifier_output = (batch_size, max_length, output_dim)
            classifier_output = self._classifier_head(last_layer_output)

        return lm_output, classifier_output
