import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from termcolor import colored

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.data_processors.tokenizers.utils import generate_from_input_ids_batch


class ConvoWizardAttentionVisualizer(object):
    def __init__(self, convo_wizard, pretrained_tokenizer, experiment_name='interpretability',
                 project_name='convo_wizard', use_cls=False, pad_token_position=0, pad_tok_type_id=0,
                 max_relative_position=None, base_path_to_save_plots=None, device=torch.device('cpu')):
        super().__init__()

        self._tokenizer = pretrained_tokenizer
        self._use_cls = use_cls
        self._cls_or_sep_idx = pretrained_tokenizer.cls_token_id if use_cls else pretrained_tokenizer.sep_token_id
        self._padding_idx = pretrained_tokenizer.pad_token_id
        self._pad_token_position = pad_token_position
        self._pad_tok_type_id = pad_tok_type_id
        self._max_relative_position = max_relative_position

        self._label_prompt_token_idx = pretrained_tokenizer.vocab['>>']
        self._awry_label_token_idx = pretrained_tokenizer.vocab['awry_label']

        self._device = device
        self._model = convo_wizard.to(device)
        self._model.eval()

        self._project_name = project_name
        self._experiment_name = experiment_name
        self._base_path_to_save_plots = base_path_to_save_plots
        self._files_setup()

    def _files_setup(self):
        if self._base_path_to_save_plots:
            self._plots_path = os.path.join(self._base_path_to_save_plots, self._project_name, self._experiment_name)
            os.makedirs(self._plots_path, exist_ok=True)

    def _get_input_ids(self, input_convo, append_label_prompt=True):
        if self._use_cls:
            input_ids = convo_tokenizer.ConvoTokenizer.tokenize(pretrained_tokenizer=self._tokenizer,
                                                                convo=input_convo, max_length=None,
                                                                pad_token_position=self._pad_token_position,
                                                                pad_tok_type_id=self._pad_tok_type_id,
                                                                labels_ignore_idx=-100)['input_ids']
        else:
            input_ids = convo_tokenizer_v2.ConvoTokenizer.tokenize(pretrained_tokenizer=self._tokenizer,
                                                                   convo=input_convo, max_length=None,
                                                                   pad_token_position=self._pad_token_position,
                                                                   pad_tok_type_id=self._pad_tok_type_id,
                                                                   labels_ignore_idx=-100)['input_ids']
        if append_label_prompt:
            return torch.hstack((torch.tensor(input_ids).squeeze(),
                                 torch.tensor([self._label_prompt_token_idx]))).unsqueeze(0).to(self._device)
        else:
            return torch.tensor(input_ids).squeeze().unsqueeze(0).to(self._device)

    @staticmethod
    def _draw(data, xticklabels, yticklabels, ax, cbar=False):
        heatmap = sns.heatmap(data, xticklabels=xticklabels, square=True, yticklabels=yticklabels, vmin=data.min(),
                              vmax=data.max(), cbar=cbar, ax=ax)
        heatmap.tick_params(labelsize=8)

    @staticmethod
    def _remove_punct(input_convo: list):
        return [re.sub(r'[.!?]', '', utt) for utt in input_convo]

    def visualize(self, input_convo: list, aggregate_at_layers=True, awry_forecast_threshold=0.644, ignore_punct=True,
                  filename_to_save_plot=None, get_intermediates=False):
        """https://aclanthology.org/W19-4808.pdf"""
        if ignore_punct:
            input_convo = self._remove_punct(input_convo=input_convo)
        convo_input_ids = self._get_input_ids(input_convo=input_convo)
        tokenized_convo = generate_from_input_ids_batch(input_ids=convo_input_ids, padding_idx=self._padding_idx,
                                                        pad_token_position=self._pad_token_position,
                                                        pad_token_type=self._pad_tok_type_id,
                                                        cls_or_sep_token_idx=self._cls_or_sep_idx,
                                                        labels_ignore_idx=-100,
                                                        max_relative_position=self._max_relative_position,
                                                        use_cls=self._use_cls, device=self._device)
        lm_output, _ = self._model(input_ids=convo_input_ids,
                                   position_ids=tokenized_convo['position_ids'],
                                   token_type_ids=tokenized_convo['token_type_ids'],
                                   attention_mask=tokenized_convo['attention_mask'], make_predictions=False)
        lm_output = lm_output[:, -1, :]
        probs = F.softmax(lm_output, dim=-1)
        awry_proba = round(probs[0][self._awry_label_token_idx].item(), 3)

        num_layers = self._model._encoder._num_encoder_layers
        xticklabels = self._tokenizer.convert_ids_to_tokens(convo_input_ids.squeeze())[:-2]  # ignore '>>', last '[SEP]'
        if not get_intermediates:
            num_plot_rows = 1 if aggregate_at_layers else num_layers
            fig, axs = plt.subplots(num_plot_rows, 1, figsize=(int(16 * len(xticklabels) / 50), 1))

        layer_attention_filters = []
        for layer in range(num_layers):
            attention_filters = self._model._encoder._encoding_layers[layer]._multi_head_attention._attention_filters
            attention_filter = attention_filters.squeeze(0).mean(dim=0).data[-1]  # last '>>' query
            attention_filter = torch.where(convo_input_ids.squeeze() == self._cls_or_sep_idx, 0.0, attention_filter)
            attention_filter = torch.concat((
                torch.tensor([0.0]),  # null attention to starting token
                attention_filter[1: -2],  # ignore the '>>' token, last '[SEP]'
            ))
            layer_attention_filters.append(attention_filter / attention_filter.sum())
            if not aggregate_at_layers and not get_intermediates:
                self._draw(attention_filter, xticklabels=xticklabels, yticklabels=[], ax=axs[layer, 0])
                if filename_to_save_plot is not None:
                    plt.savefig(os.path.join(self._plots_path, f'layer_{layer}_{filename_to_save_plot}'), dpi=300)
                plt.show()
        if aggregate_at_layers and not get_intermediates:
            self._draw(torch.stack(layer_attention_filters).mean(dim=0).unsqueeze(0), xticklabels=xticklabels,
                       yticklabels=[], ax=axs)
            if filename_to_save_plot is not None:
                plt.savefig(os.path.join(self._plots_path, f'{filename_to_save_plot}'), dpi=300)
            plt.show()

        forecast = colored('awry', 'red') if awry_proba >= awry_forecast_threshold else colored('calm', 'green')
        print(f'forecast: {forecast}')
        return awry_proba if not get_intermediates else (awry_proba, xticklabels,
                                                         torch.stack(layer_attention_filters).mean(dim=0).unsqueeze(0))
