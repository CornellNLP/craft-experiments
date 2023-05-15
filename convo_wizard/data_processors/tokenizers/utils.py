from itertools import chain

import numpy as np
import torch

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2


def batch_tokenize(data_instances, pretrained_tokenizer, max_length=2048, pad_token_position=0, pad_tok_type_id=0,
                   labels_ignore_idx=-100, use_sep=False, label_by_cls_mask=False):
    tokenized_convos = {'input_ids': [], 'position_ids': [], 'attention_mask': [], 'cls_mask': [], 'token_type_ids': [],
                        'relative_position_ids': []}

    labels = None
    try:
        labels = data_instances['label']
        tokenized_convos['labels'] = []
    except KeyError:
        pass

    for convo_idx, convo in enumerate(data_instances['convo']):
        # `padding='max_length'` vs. `padding=True` (batched padding).
        if use_sep:
            tokenized_convo = convo_tokenizer.ConvoTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer,
                                                                      convo=convo, max_length=max_length,
                                                                      pad_token_position=pad_token_position,
                                                                      pad_tok_type_id=pad_tok_type_id,
                                                                      labels_ignore_idx=labels_ignore_idx)
        else:
            tokenized_convo = convo_tokenizer_v2.ConvoTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer,
                                                                         convo=convo, max_length=max_length,
                                                                         pad_token_position=pad_token_position,
                                                                         pad_tok_type_id=pad_tok_type_id,
                                                                         labels_ignore_idx=labels_ignore_idx)

        tokenized_convos['input_ids'].append(tokenized_convo['input_ids'])
        tokenized_convos['position_ids'].append(tokenized_convo['position_ids'])
        tokenized_convos['attention_mask'].append(tokenized_convo['attention_mask'])
        tokenized_convos['cls_mask'].append(tokenized_convo['cls_mask'])
        tokenized_convos['token_type_ids'].append(tokenized_convo['token_type_ids'])
        tokenized_convos['relative_position_ids'].append(tokenized_convo['relative_position_ids'])

        if labels is not None:
            if label_by_cls_mask:
                tokenized_convo_labels = tokenized_convo['cls_mask']  # -100 at non-CLS, 0 at CLS tokens
                if labels[convo_idx] != 0:
                    sent_cls_token_idxs = np.where(tokenized_convo['cls_mask'] == 0)[0]
                    for sent_idx in sent_cls_token_idxs:
                        tokenized_convo_labels[sent_idx] = int(labels[convo_idx])  # mark all CLS heads with predictions
            else:
                tokenized_convo_labels = np.array([-100] * len(tokenized_convo['cls_mask']))
                tokenized_convo_labels[0] = int(labels[convo_idx])  # use only the first CLS head to make predictions
            tokenized_convos['labels'].append(tokenized_convo_labels)

    return tokenized_convos


def generate_from_input_ids_batch(input_ids, padding_idx=0, pad_token_position=0, pad_token_type=0, cls_token_idx=2,
                                  labels_ignore_idx=-100, max_relative_position=None, device=torch.device('cpu')):
    assert len(input_ids.shape) == 2
    batch_size, input_len = input_ids.shape[0], input_ids.shape[1]

    position_ids = torch.empty(size=input_ids.shape, device=device)
    if max_relative_position is None:
        position_ids = 1 + torch.arange(input_len).expand(batch_size, -1)

    cls_mask = torch.where(input_ids == cls_token_idx, 0, labels_ignore_idx)
    segment_ids = torch.empty(size=input_ids.shape, device=device)
    for idx in range(batch_size):
        cls_idxs = torch.cat((torch.where(cls_mask[idx, :] == 0)[0].to(device),
                              torch.tensor([input_len], device=device)))
        _segment_ids = [[int(idx % 2 != 0) + 1] * (cls_idxs[idx + 1] - cls_idxs[idx]) for idx in
                        range(len(cls_idxs) - 1)]
        segment_ids[idx, :] = torch.tensor(list(chain.from_iterable(_segment_ids)), device=device)

        if max_relative_position is not None:
            _relative_position_ids = [1 + torch.arange(cls_idxs[_ + 1] - cls_idxs[_]) for _ in range(len(cls_idxs) - 1)]
            position_ids[idx, :] = torch.tensor(list(chain.from_iterable(_relative_position_ids)), device=device)

    segment_ids[input_ids == padding_idx] = pad_token_type
    position_ids[input_ids == padding_idx] = pad_token_position

    return {'position_ids': position_ids,
            'attention_mask': torch.where(input_ids == padding_idx, 1, 0),
            'token_type_ids': segment_ids}
