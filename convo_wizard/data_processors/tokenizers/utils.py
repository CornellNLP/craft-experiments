from itertools import chain

import torch

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer


def batch_tokenize(data_instances, pretrained_tokenizer, max_length=2048, pad_token_position=0, pad_tok_type_id=0,
                   labels_ignore_idx=-100):
    tokenized_convos = {'input_ids': [], 'position_ids': [], 'attention_mask': [], 'cls_mask': [], 'token_type_ids': [],
                        'relative_position_ids': []}

    for convo in data_instances['convos']:
        # `padding='max_length'` vs. `padding=True` (batched padding).
        tokenized_convo = ConvoTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer, convo=convo,
                                                  max_length=max_length, pad_token_position=pad_token_position,
                                                  pad_tok_type_id=pad_tok_type_id, labels_ignore_idx=labels_ignore_idx)

        tokenized_convos['input_ids'].append(tokenized_convo['input_ids'])
        tokenized_convos['position_ids'].append(tokenized_convo['position_ids'])
        tokenized_convos['attention_mask'].append(tokenized_convo['attention_mask'])
        tokenized_convos['cls_mask'].append(tokenized_convo['cls_mask'])
        tokenized_convos['token_type_ids'].append(tokenized_convo['token_type_ids'])
        tokenized_convos['relative_position_ids'].append(tokenized_convo['relative_position_ids'])

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
        cls_idxs = torch.cat((torch.where(cls_mask[idx, :] == 0)[0], torch.tensor([input_len])))
        _segment_ids = [[int(idx % 2 != 0)] * (cls_idxs[idx + 1] - cls_idxs[idx]) for idx in range(len(cls_idxs) - 1)]
        segment_ids[idx, :] = torch.tensor(list(chain.from_iterable(_segment_ids)))

        if max_relative_position is not None:
            _relative_position_ids = [1 + torch.arange(cls_idxs[_ + 1] - cls_idxs[_]) for _ in range(len(cls_idxs) - 1)]
            position_ids[idx, :] = torch.tensor(list(chain.from_iterable(_relative_position_ids)))

    segment_ids[input_ids == padding_idx] = pad_token_type
    position_ids[input_ids == padding_idx] = pad_token_position

    return {'position_ids': position_ids,
            'attention_mask': torch.where(input_ids == padding_idx, 1, 0),
            'cls_mask': cls_mask,
            'token_type_ids': segment_ids}
