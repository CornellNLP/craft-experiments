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
