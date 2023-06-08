from collections import namedtuple
from itertools import chain

import numpy as np
import torch

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2


def _populate_labels(label, cls_or_sep_mask, use_cls=False, label_at_each_utt=False):
    tokenized_convo_labels = np.array([-100] * len(cls_or_sep_mask))
    sent_cls_or_sep_token_idxs = np.where(cls_or_sep_mask == 0)[0]  # -100 at non-CLS/SEP, 0 at CLS/SEP tokens
    if label_at_each_utt:
        for sent_idx in sent_cls_or_sep_token_idxs:
            tokenized_convo_labels[sent_idx] = int(label)
    else:
        # Use only the first [CLS] or last [SEP] token make predictions.
        if use_cls:
            tokenized_convo_labels[sent_cls_or_sep_token_idxs[0]] = int(label)
        else:
            tokenized_convo_labels[sent_cls_or_sep_token_idxs[-1]] = int(label)
    return tokenized_convo_labels


def _tokenize(pretrained_tokenizer, convo, is_label_appended, is_label_prepended, use_cls, max_length=None,
              pad_token_position=0, pad_tok_type_id=0, labels_ignore_idx=-100):
    if use_cls:
        return convo_tokenizer.ConvoTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer, convo=convo,
                                                       max_length=max_length, is_label_appended=is_label_appended,
                                                       is_label_prepended=is_label_prepended,
                                                       pad_token_position=pad_token_position,
                                                       pad_tok_type_id=pad_tok_type_id,
                                                       labels_ignore_idx=labels_ignore_idx)
    else:
        return convo_tokenizer_v2.ConvoTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer, convo=convo,
                                                          max_length=max_length, is_label_appended=is_label_appended,
                                                          is_label_prepended=is_label_prepended,
                                                          pad_token_position=pad_token_position,
                                                          pad_tok_type_id=pad_tok_type_id,
                                                          labels_ignore_idx=labels_ignore_idx)


def batch_tokenize(data_instances, pretrained_tokenizer, max_length=2048, pad_token_position=0, pad_tok_type_id=0,
                   labels_ignore_idx=-100, use_cls=False, label_at_each_utt=False, append_label=False,
                   prepend_label=False, lm_prompt_break_at_utt=False, lm_label_break_at_utt=False,
                   lm_also_include_convo=False):
    tokenized_convos = {'input_ids': [], 'position_ids': [], 'attention_mask': [], 'cls_mask': [], 'sep_mask': [],
                        'token_type_ids': [], 'relative_position_ids': []}

    labels = None
    try:
        labels = data_instances['label']
        tokenized_convos['labels'] = []
    except KeyError:
        pass

    _Convo = namedtuple('_Convo', ['convo', 'is_label_appended', 'is_label_prepended'])

    for convo_idx, convo in enumerate(data_instances['convo']):
        to_process_queue = []
        if not (append_label or prepend_label) or labels is None or lm_also_include_convo:
            if lm_prompt_break_at_utt:
                utt_context = []
                for utt in convo:
                    utt_context = utt_context + [utt]
                    to_process_queue.append(_Convo(convo=utt_context, is_label_appended=False,
                                                   is_label_prepended=False))
            else:
                to_process_queue.append(_Convo(convo=convo, is_label_appended=False, is_label_prepended=False))

        # Label appended/prepended to the sequence: https://arxiv.org/pdf/1912.10165.pdf.
        if labels is not None:
            if append_label:
                answer = f' >> {"awry_label" if int(labels[convo_idx]) == 1 else "calm_label"}'
                if lm_label_break_at_utt:
                    utt_context = []  # add one utt per dataset entry
                    for utt in convo:
                        appended_convo = utt_context + [utt] + [answer]
                        to_process_queue.append(_Convo(convo=appended_convo, is_label_appended=True,
                                                       is_label_prepended=False))
                        utt_context = utt_context + [utt]
                else:
                    appended_convo = convo + [answer]
                    to_process_queue.append(_Convo(convo=appended_convo, is_label_appended=True,
                                                   is_label_prepended=False))
            if prepend_label:
                prompt = f'{"awry_prompt" if int(labels[convo_idx]) == 1 else "calm_prompt"} << '
                if lm_prompt_break_at_utt:
                    utt_context = []  # add one utt per dataset entry
                    for utt in convo:
                        prepended_convo = [prompt] + utt_context + [utt]
                        to_process_queue.append(_Convo(convo=prepended_convo, is_label_appended=False,
                                                       is_label_prepended=True))
                        utt_context = utt_context + [utt]
                else:
                    prepended_convo = [prompt] + convo
                    to_process_queue.append(_Convo(convo=prepended_convo, is_label_appended=False,
                                                   is_label_prepended=True))

        for _convo in to_process_queue:
            # `padding='max_length'` vs. `padding=True` (batched padding).
            tokenized_convo = _tokenize(pretrained_tokenizer=pretrained_tokenizer, convo=_convo.convo,
                                        is_label_appended=_convo.is_label_appended,
                                        is_label_prepended=_convo.is_label_prepended, use_cls=use_cls,
                                        max_length=max_length, pad_token_position=pad_token_position,
                                        pad_tok_type_id=pad_tok_type_id, labels_ignore_idx=labels_ignore_idx)

            tokenized_convos['input_ids'].append(tokenized_convo['input_ids'])
            tokenized_convos['position_ids'].append(tokenized_convo['position_ids'])
            tokenized_convos['attention_mask'].append(tokenized_convo['attention_mask'])
            tokenized_convos['cls_mask'].append(tokenized_convo['cls_mask'])
            tokenized_convos['sep_mask'].append(tokenized_convo['sep_mask'])
            tokenized_convos['token_type_ids'].append(tokenized_convo['token_type_ids'])
            tokenized_convos['relative_position_ids'].append(tokenized_convo['relative_position_ids'])

            # Use the last token to make predictions.
            # TODO: other options: https://github.com/huggingface/transformers/issues/3168#issuecomment-697263861.
            if labels is not None:
                cls_or_sep_mask = tokenized_convo['cls_mask'] if use_cls else tokenized_convo['sep_mask']
                tokenized_convo_labels = _populate_labels(label=labels[convo_idx], cls_or_sep_mask=cls_or_sep_mask,
                                                          use_cls=use_cls, label_at_each_utt=label_at_each_utt)
                tokenized_convos['labels'].append(tokenized_convo_labels)

    return tokenized_convos


def generate_from_input_ids_batch(input_ids, padding_idx=0, pad_token_position=0, pad_token_type=0,
                                  cls_or_sep_token_idx=2, labels_ignore_idx=-100, max_relative_position=None,
                                  use_cls=False, device=torch.device('cpu')):
    assert len(input_ids.shape) == 2
    batch_size, input_len = input_ids.shape[0], input_ids.shape[1]

    position_ids = torch.empty(size=input_ids.shape, device=device)
    if max_relative_position is None:
        position_ids = 1 + torch.arange(input_len).expand(batch_size, -1)

    cls_or_sep_mask = torch.where(input_ids == cls_or_sep_token_idx, 0, labels_ignore_idx)
    segment_ids = torch.empty(size=input_ids.shape, device=device)
    for idx in range(batch_size):
        if use_cls:
            cls_or_sep_idxs = torch.cat((torch.where(cls_or_sep_mask[idx, :] == 0)[0].to(device),
                                         torch.tensor([input_len], device=device)))
        else:
            cls_or_sep_idxs = torch.cat((torch.tensor([-1], device=device),
                                         torch.where(cls_or_sep_mask[idx, :] == 0)[0].to(device),
                                         torch.tensor([input_len - 1], device=device)))
        _segment_ids = [[int(idx % 2 != 0) + 1] * (cls_or_sep_idxs[idx + 1] - cls_or_sep_idxs[idx]) for idx in
                        range(len(cls_or_sep_idxs) - 1)]
        segment_ids[idx, :] = torch.tensor(list(chain.from_iterable(_segment_ids)), device=device)

        if max_relative_position is not None:
            if use_cls:
                _relative_position_ids = [1 + torch.arange(cls_or_sep_idxs[_ + 1] - cls_or_sep_idxs[_]) for _ in
                                          range(len(cls_or_sep_idxs) - 1)]
            else:
                _relative_position_ids = [1 + torch.arange(cls_or_sep_idxs[_ + 1] - cls_or_sep_idxs[_])[::-1] for _ in
                                          range(len(cls_or_sep_idxs) - 1)]
            position_ids[idx, :] = torch.tensor(list(chain.from_iterable(_relative_position_ids)), device=device)

    segment_ids[input_ids == padding_idx] = pad_token_type
    position_ids[input_ids == padding_idx] = pad_token_position

    return {'position_ids': position_ids,
            'attention_mask': torch.where(input_ids == padding_idx, 1, 0),
            'token_type_ids': segment_ids}
