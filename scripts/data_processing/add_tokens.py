from argparse import ArgumentParser

import torch
import yaml

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.models.convo_wizard import ConvoWizard


def main(config_path, model_config_path, tokenizer_path, pretrained_checkpoint_path=None, pretrained_model_path=None):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    with open(model_config_path, 'r') as fp:
        model_config = yaml.safe_load(fp)

    if config['weak_supervision_tokens']['use_cls']:
        convo_uncased_tokenizer = convo_tokenizer.ConvoTokenizer.load(tokenizer_path)
        cls_or_sep_token_idx = convo_uncased_tokenizer.cls_token_id
    else:
        convo_uncased_tokenizer = convo_tokenizer_v2.ConvoTokenizer.load(tokenizer_path)
        cls_or_sep_token_idx = convo_uncased_tokenizer.sep_token_id
    old_vocab_size = convo_uncased_tokenizer.vocab_size
    num_tokens_added = convo_uncased_tokenizer.add_tokens(**config['weak_supervision_tokens']['args'])
    if tokenizer_path[-1] == '/':
        tokenizer_path = tokenizer_path[:-1]
    convo_uncased_tokenizer.save_pretrained(f'{tokenizer_path}.vocab_resized')

    device = model_config['general']['device']
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    convo_wizard = ConvoWizard(vocab_size=old_vocab_size, padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_or_sep_token_idx=cls_or_sep_token_idx, device=device,
                               **model_config['transformer']['args'])
    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device.type)
        convo_wizard.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_model_path is not None:
        convo_wizard.from_pretrained(model_path=pretrained_model_path)
    convo_wizard._encoder._embedding._token_embedding.resize_token_embedding(num_new_tokens=num_tokens_added)

    if pretrained_checkpoint_path is not None:
        checkpoint['model_state_dict'] = convo_wizard.state_dict()
        torch.save(checkpoint, f'{pretrained_checkpoint_path}.vocab_resized')
    elif pretrained_model_path is not None:
        convo_wizard.save_pretrained(f'{pretrained_model_path}.vocab_resized')


if __name__ == '__main__':
    parser = ArgumentParser(description='add special tokens to the end of the vocabulary and end of token embeddings')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--model_config_path', type=str, help='path to the config file of convo-wizard model')
    parser.add_argument('--tokenizer_path', type=str, help='path to load the tokenizer from')
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)

    args = parser.parse_args()

    main(config_path=args.config_path, model_config_path=args.model_config_path, tokenizer_path=args.tokenizer_path,
         pretrained_checkpoint_path=args.pretrained_checkpoint_path, pretrained_model_path=args.pretrained_model_path)
