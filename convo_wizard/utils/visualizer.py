import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.utils.utils import device_mapper


class ConvoWizardAttentionVisualizer(object):
    def __init__(self, convo_wizard, pretrained_tokenizer, experiment_name, project_name='convo_wizard', use_cls=False,
                 pad_token_position=0, pad_tok_type_id=0, max_relative_position=None, base_path_to_save_plots=None):
        super().__init__()

        self._tokenizer = pretrained_tokenizer
        self._use_cls = use_cls
        self._pad_token_position = pad_token_position
        self._pad_tok_type_id = pad_tok_type_id
        self._use_relative_position_ids = True if max_relative_position is not None else False

        self._model = convo_wizard

        self._project_name = project_name
        self._experiment_name = experiment_name
        self._base_path_to_save_plots = base_path_to_save_plots
        self._files_setup()

    def _files_setup(self):
        self._plots_path = os.path.join(self._base_path_to_save_plots, self._project_name, self._experiment_name)
        os.makedirs(self._plots_path, exist_ok=True)

    def _tokenize(self, input_convo):
        if self._use_cls:
            tokenized_convo = convo_tokenizer.ConvoTokenizer.tokenize(pretrained_tokenizer=self._tokenizer,
                                                                      convo=input_convo, max_length=None,
                                                                      pad_token_position=self._pad_token_position,
                                                                      pad_tok_type_id=self._pad_tok_type_id,
                                                                      labels_ignore_idx=-100)
        else:
            tokenized_convo = convo_tokenizer_v2.ConvoTokenizer.tokenize(pretrained_tokenizer=self._tokenizer,
                                                                         convo=input_convo, max_length=None,
                                                                         pad_token_position=self._pad_token_position,
                                                                         pad_tok_type_id=self._pad_tok_type_id,
                                                                         labels_ignore_idx=-100)
        if self._use_relative_position_ids:
            position_ids = torch.tensor(tokenized_convo['relative_position_ids']).unsqueeze(0)
        else:
            position_ids = torch.tensor(tokenized_convo['position_ids']).unsqueeze(0)
        return {'input_ids': torch.tensor(tokenized_convo['input_ids']).unsqueeze(0), 'position_ids': position_ids,
                'token_type_ids': torch.tensor(tokenized_convo['token_type_ids']).unsqueeze(0),
                'attention_mask': torch.tensor(tokenized_convo['attention_mask']).unsqueeze(0),
                'sep_mask': torch.tensor(tokenized_convo['sep_mask'])}

    @staticmethod
    def _draw(data, xticklabels, yticklabels, ax, cbar_ax):
        heatmap = sns.heatmap(data, xticklabels=xticklabels, square=True, yticklabels=yticklabels, vmin=data.min(),
                              vmax=data.max(), cbar=True, cbar_ax=cbar_ax, ax=ax)
        heatmap.tick_params(labelsize=10)

    def visualize(self, input_convo: list, num_tokens=35, visualization_start_idx=0, layers_to_plot=None,
                  filename_to_save_plot=None):
        tokenized_convo = self._tokenize(input_convo=input_convo)
        _, classifier_output = self._model(input_ids=tokenized_convo['input_ids'],
                                           position_ids=tokenized_convo['position_ids'],
                                           token_type_ids=tokenized_convo['token_type_ids'],
                                           attention_mask=tokenized_convo['attention_mask'], make_predictions=True)
        preds = classifier_output.detach().view(-1, classifier_output.shape[-1]).softmax(dim=-1)[:, -1].numpy()
        forecast_proba = np.max(np.where(tokenized_convo['sep_mask'] != -100, preds, -np.inf))

        inputs = self._tokenizer.decode(tokenized_convo['input_ids'].squeeze()).split()
        visualization_start_idx = min(visualization_start_idx, len(inputs))
        end_idx = visualization_start_idx + min(len(inputs), num_tokens)
        if layers_to_plot is None:
            layers_to_plot = [self._model._encoder._num_encoder_layers - 1]

        for layer_num in layers_to_plot:
            num_heads = self._model._encoder._encoding_layers[0]._multi_head_attention._num_heads

            num_plt_cols = 3
            num_plt_rows = 1 + ((num_heads - 1) // num_plt_cols)
            plt_height = min(9, 2 + len(inputs) // 2) * num_plt_rows
            fig, axs = plt.subplots(num_plt_rows, num_plt_cols, figsize=(25, plt_height))

            cbar_ax = fig.add_axes([0.905, 0.3, 0.005, 0.3])
            attention_filters = \
                self._model._encoder._encoding_layers[layer_num]._multi_head_attention._attention_filters.squeeze(0)
            for head_num in range(num_heads):
                attention_filter = device_mapper(attention_filters[head_num], torch.device('cpu')).data
                attention_filter = attention_filter[visualization_start_idx: end_idx, visualization_start_idx: end_idx]

                ticklabels = inputs[visualization_start_idx: end_idx]
                ax = axs[head_num // num_plt_cols, head_num % num_plt_cols] if num_plt_rows > 1 else axs[
                    head_num % num_plt_cols]
                ax.set_title(f'layer: {layer_num}, head: {head_num}')
                self._draw(data=attention_filter, xticklabels=ticklabels, yticklabels=ticklabels, ax=ax,
                           cbar_ax=cbar_ax)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)

            if filename_to_save_plot is not None:
                plt.savefig(os.path.join(self._plots_path, f'layer_{layer_num}_{filename_to_save_plot}'), dpi=300)
            plt.show()
        return forecast_proba
