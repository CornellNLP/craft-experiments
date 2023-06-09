import os
from argparse import ArgumentParser

import torch
import yaml

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.utils.utils import set_seed


def main(prompt_convo, config_path, tokenizer_path, pretrained_checkpoint_path, pretrained_model_path=None,
         utt_separator='<|endofutt|>'):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    device = config['general']['device']
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if config['tokenizer']['use_cls']:
        convo_uncased_tokenizer = convo_tokenizer.ConvoTokenizer.load(tokenizer_path)
        cls_or_sep_token_idx = convo_uncased_tokenizer.cls_token_id
    else:
        convo_uncased_tokenizer = convo_tokenizer_v2.ConvoTokenizer.load(tokenizer_path)
        cls_or_sep_token_idx = convo_uncased_tokenizer.sep_token_id
    convo_wizard = ConvoWizard(vocab_size=len(convo_uncased_tokenizer),
                               padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_or_sep_token_idx=cls_or_sep_token_idx, device=device,
                               **config['transformer']['args'])

    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device.type)
        convo_wizard.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_model_path is not None:
        convo_wizard.from_pretrained(model_path=pretrained_model_path)

    if config['tokenizer']['use_cls']:
        tokenized_convo = \
            convo_tokenizer.ConvoTokenizer.tokenize(pretrained_tokenizer=convo_uncased_tokenizer,
                                                    convo=list(map(str.strip, prompt_convo.split(utt_separator))),
                                                    max_length=None)
    else:
        tokenized_convo = \
            convo_tokenizer_v2.ConvoTokenizer.tokenize(pretrained_tokenizer=convo_uncased_tokenizer,
                                                       convo=list(map(str.strip, prompt_convo.split(utt_separator))),
                                                       max_length=None)
    # Ignore the last '[SEP]' token added at the end of the sequence by the tokenizer.
    input_ids = torch.tensor(tokenized_convo['input_ids'][:-1]).expand(config['generate']['args']['num_samples'], -1)
    augmented_input_ids = convo_wizard.generate(input_ids=input_ids, use_cls=config['tokenizer']['use_cls'],
                                                **config['generate']['args'])
    for sample_idx in range(config['generate']['args']['num_samples']):
        print(convo_uncased_tokenizer.decode(augmented_input_ids[sample_idx].cpu().squeeze()))
        print('-' * 80)


if __name__ == '__main__':
    parser = ArgumentParser(description='use the trained LM to generate text')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--prompt_convo', type=str, help='the prompt conversation, separated by utt_separator')
    parser.add_argument('--utt_separator', type=str, help='the utterance separator', default='<|endofutt|>')

    args = parser.parse_args()

    main(config_path=args.config_path, tokenizer_path=args.tokenizer_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         prompt_convo=args.prompt_convo, utt_separator=args.utt_separator)
