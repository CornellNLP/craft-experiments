import os
from argparse import ArgumentParser

import torch
import yaml

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.utils.utils import set_seed
from convo_wizard.utils.visualizer import ConvoWizardAttentionVisualizer


def main(input_convo, config_path, tokenizer_path, pretrained_checkpoint_path, pretrained_model_path=None,
         num_tokens=35, visualization_start_idx=0, layers_to_plot=None, filename_to_save_plot=None,
         base_path_to_save_plots=None, utt_separator='<|endofutt|>', experiment_name='experiment',
         project_name='convo_wizard'):
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

    convo_wizard = ConvoWizard(vocab_size=convo_uncased_tokenizer.vocab_size,
                               padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_or_sep_token_idx=cls_or_sep_token_idx, device=device,
                               **config['transformer']['args'])
    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device.type)
        convo_wizard.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_model_path is not None:
        convo_wizard.from_pretrained(model_path=pretrained_model_path)

    convo_visualizer = \
        ConvoWizardAttentionVisualizer(convo_wizard=convo_wizard, pretrained_tokenizer=convo_uncased_tokenizer,
                                       experiment_name=experiment_name, project_name=project_name,
                                       use_cls=config['tokenizer']['use_cls'],
                                       pad_token_position=config['transformer']['args']['pad_token_position'],
                                       pad_tok_type_id=config['transformer']['args']['pad_tok_type'],
                                       max_relative_position=config['transformer']['args']['max_relative_position'],
                                       base_path_to_save_plots=base_path_to_save_plots)
    forecast_proba = convo_visualizer.visualize(input_convo=list(map(str.strip, input_convo.split(utt_separator))),
                                                num_tokens=num_tokens, visualization_start_idx=visualization_start_idx,
                                                layers_to_plot=layers_to_plot,
                                                filename_to_save_plot=filename_to_save_plot)
    print(f'toxicity forecast proba: {forecast_proba}')


if __name__ == '__main__':
    parser = ArgumentParser(description='use the trained LM to visualize attention plots')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--base_path_to_save_plots', type=str, help='base path to store plots', default=os.getcwd())
    parser.add_argument('--experiment_name', type=str, help='the experiment name', default='convo_wizard_experiment')
    parser.add_argument('--project_name', type=str, help='the project name', default='convo_wizard')
    parser.add_argument('--num_tokens', type=int, help='the number of tokens to show on the attention plot', default=35)
    parser.add_argument('--visualization_start_idx', type=int, help='the start token index', default=0)
    parser.add_argument('--layers_to_plot', nargs='*', type=int, help='the layers to plot', default=[])
    parser.add_argument('--input_convo', type=str, help='the input conversation, separated by utt_separator')
    parser.add_argument('--utt_separator', type=str, help='the utterance separator', default='<|endofutt|>')
    parser.add_argument('--filename_to_save_plot', type=str, help='the filename to save the plot', default=None)

    args = parser.parse_args()

    main(config_path=args.config_path, tokenizer_path=args.tokenizer_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         base_path_to_save_plots=args.base_path_to_save_plots, experiment_name=args.experiment_name,
         project_name=args.project_name, num_tokens=args.num_tokens,
         visualization_start_idx=args.visualization_start_idx, layers_to_plot=args.layers_to_plot,
         input_convo=args.input_convo, utt_separator=args.utt_separator,
         filename_to_save_plot=args.filename_to_save_plot)
