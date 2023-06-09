import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import autograd

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
        self._calm_label_token_idx = pretrained_tokenizer.vocab['calm_label']

        self._device = device
        self._model = convo_wizard.to(device)
        self._model.eval()

        self._hooks = {'hooks': {}, 'outputs': {}}
        self._attach_hook(module=self._model._encoder._embedding, hook=self.__get_embeddings_hook)

        self._project_name = project_name
        self._experiment_name = experiment_name
        self._base_path_to_save_plots = base_path_to_save_plots
        self._files_setup()

    def _files_setup(self):
        if self._base_path_to_save_plots:
            self._plots_path = os.path.join(self._base_path_to_save_plots, self._project_name, self._experiment_name)
            os.makedirs(self._plots_path, exist_ok=True)

    def __get_embeddings_hook(self, module, inputs, embeddings):
        self._hooks['outputs']['input_embeddings'] = embeddings

    def _attach_hook(self, module, hook):
        self._hooks['hooks'][hook.__name__] = module.register_forward_hook(hook)

    def _detach_all_hooks(self):
        for handle in self._hooks['hooks'].values():
            handle.remove()
        self._hooks = {'hooks': {}, 'outputs': {}}

    def _detach_hook(self, hook):
        self._hooks['hooks'][hook.__name__].remove()
        self._hooks['hooks'].pop(hook.__name__)

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

    def saliency(self, input_convo: list, ignore_punct=True, get_intermediates=True):
        """input x gradient: https://jalammar.github.io/explaining-transformers/."""
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
        lm_output, _ = self._model(input_ids=convo_input_ids, position_ids=tokenized_convo['position_ids'],
                                   token_type_ids=tokenized_convo['token_type_ids'],
                                   attention_mask=tokenized_convo['attention_mask'], make_predictions=False)
        lm_output = lm_output[:, -1, :]
        probs = F.softmax(lm_output, dim=-1)
        awry_proba = round(probs[0][self._awry_label_token_idx].item(), 3)
        calm_proba = round(probs[0][self._calm_label_token_idx].item(), 3)
        label_token_idx = self._awry_label_token_idx if awry_proba >= calm_proba else self._calm_label_token_idx

        self._model.zero_grad(set_to_none=True)
        gradients = autograd.grad(lm_output[0, label_token_idx], self._hooks['outputs']['input_embeddings'],
                                  retain_graph=False, create_graph=False)[0].detach().cpu()
        input_embeddings = self._hooks['outputs']['input_embeddings'].detach().cpu()
        saliency_attributions = torch.norm((gradients * input_embeddings).squeeze(), dim=1)
        saliency_attributions = saliency_attributions / torch.sum(saliency_attributions)
        if not get_intermediates:
            return saliency_attributions
        input_tokens = self._tokenizer.convert_ids_to_tokens(convo_input_ids.squeeze())[:-2]
        return awry_proba, calm_proba, input_tokens, saliency_attributions

    def integrated_gradients(self, input_convo: list, ignore_punct=True, get_intermediates=True, num_steps=50):
        """https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py."""
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
        input_embeddings = self._model._encoder._embedding(input_ids=convo_input_ids,
                                                           position_ids=tokenized_convo['position_ids'],
                                                           token_type_ids=tokenized_convo['token_type_ids'])
        lm_output, _ = self._model(input_embeddings=input_embeddings, make_predictions=False)
        probs = F.softmax(lm_output[:, -1, :], dim=-1)
        awry_proba = round(probs[0][self._awry_label_token_idx].item(), 3)
        calm_proba = round(probs[0][self._calm_label_token_idx].item(), 3)
        label_token_idx = self._awry_label_token_idx if awry_proba >= calm_proba else self._calm_label_token_idx

        baseline_input_ids = torch.where(convo_input_ids != self._label_prompt_token_idx, self._padding_idx,
                                         convo_input_ids)
        baseline_tokenized_convo = generate_from_input_ids_batch(input_ids=baseline_input_ids,
                                                                 padding_idx=self._padding_idx,
                                                                 pad_token_position=self._pad_token_position,
                                                                 pad_token_type=self._pad_tok_type_id,
                                                                 cls_or_sep_token_idx=self._cls_or_sep_idx,
                                                                 labels_ignore_idx=-100,
                                                                 max_relative_position=self._max_relative_position,
                                                                 use_cls=self._use_cls, device=self._device)
        baseline_embeddings = self._model._encoder._embedding(input_ids=baseline_input_ids,
                                                              position_ids=baseline_tokenized_convo['position_ids'],
                                                              token_type_ids=baseline_tokenized_convo['token_type_ids'])

        input_embeddings = input_embeddings.detach().cpu()
        baseline_embeddings = baseline_embeddings.detach().cpu()
        # https://towardsdatascience.com/integrated-gradients-from-scratch-b46311e4ab4.
        scaled_embeddings = \
            torch.cat([baseline_embeddings + (float(_) / num_steps) * (input_embeddings - baseline_embeddings)
                       for _ in range(0, num_steps + 1)], dim=0).requires_grad_()

        self._model.zero_grad(set_to_none=True)
        lm_output, _ = self._model(input_embeddings=scaled_embeddings, make_predictions=False)
        pred_probs = F.softmax(lm_output[:, -1, :], dim=-1)[:, label_token_idx]
        gradients = autograd.grad(torch.unbind(pred_probs), scaled_embeddings)[0].detach().cpu().numpy()

        # Using trapezoidal approximation of integral: https://arxiv.org/abs/1908.06214.
        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_grads = (input_embeddings - baseline_embeddings) * np.average(gradients, axis=0)
        attributions = torch.norm(integrated_grads.squeeze(), dim=1)
        attributions = attributions / torch.sum(attributions)
        if not get_intermediates:
            return attributions
        input_tokens = self._tokenizer.convert_ids_to_tokens(convo_input_ids.squeeze())[:-2]
        return awry_proba, calm_proba, input_tokens, attributions

    def visualize(self, input_convo: list, aggregate_at_layers=True, awry_forecast_threshold=0.644, ignore_punct=True,
                  filename_to_save_plot=None, get_intermediates=False):
        """https://aclanthology.org/W19-4808.pdf."""
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
        calm_proba = round(probs[0][self._calm_label_token_idx].item(), 3)

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
            try:
                dampen_last_token_by = 2 if not xticklabels[-1] in string.punctuation else 10
                attention_filter = torch.concat((
                    torch.tensor([0.0]),  # null attention to starting token
                    attention_filter[1: -3],  # ignore the '>>' token, last '[SEP]', and the last possibly punct token
                    (attention_filter[-3] / dampen_last_token_by).unsqueeze(0),  # dampened last token attention
                ))
            except IndexError:
                attention_filter = torch.concat((torch.tensor([0.0]), attention_filter[1: -2]))
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

        if not get_intermediates:
            forecast = colored('awry', 'red') if awry_proba >= awry_forecast_threshold else colored('calm', 'green')
            print(f'forecast: {forecast}')
            return awry_proba
        attention_scores = torch.mean(torch.stack(layer_attention_filters), dim=0)
        attention_scores = attention_scores / torch.sum(attention_scores)  # https://stackoverflow.com/a/52223289
        return awry_proba, calm_proba, xticklabels, attention_scores
