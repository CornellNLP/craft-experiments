import os
from argparse import ArgumentParser

import datasets
import torch
import yaml
from torch import nn
from torch.optim import Adam

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.optimizers.noam import NoamOptimizer
from convo_wizard.trainers.trainer import ConvoWizardTrainer
from convo_wizard.utils.tracker import Tracker
from convo_wizard.utils.utils import set_seed


def main(config_path, base_path_to_store_results, tokenizer_path, train_data_path, pretrained_model_path=None,
         pretrained_checkpoint_path=None, experiment_name='experiment', project_name='convo_wizard',
         entity_name='cornell-nlp', log_to_wandb=True):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    tracker = Tracker(config=config, base_path_to_store_results=base_path_to_store_results,
                      experiment_name=experiment_name, project_name=project_name, entity_name=entity_name,
                      log_to_wandb=log_to_wandb)

    device = config['general']['device']
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    convo_uncased_tokenizer = ConvoTokenizer.load(tokenizer_path)
    tokenized_train_data = datasets.load_dataset('json', data_files={'train': train_data_path})['train']

    convo_wizard = ConvoWizard(vocab_size=convo_uncased_tokenizer.vocab_size,
                               padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_token_idx=convo_uncased_tokenizer.cls_token_id, device=device,
                               **config['transformer']['args'])
    optimizer = NoamOptimizer(Adam(convo_wizard.get_trainable_params(), **config['optimizer']['adam']['args']),
                              embedding_dim=config['transformer']['args']['embedding_dim'],
                              **config['optimizer']['args'])
    trainer = ConvoWizardTrainer(convo_wizard=convo_wizard, optimizer=optimizer, tracker=tracker,
                                 tokenized_train_data=tokenized_train_data, tokenized_val_data=None,
                                 loss_fn=nn.CrossEntropyLoss, device=device, **config['trainer']['args']['generator'])

    if pretrained_checkpoint_path is not None:
        trainer.load_from_checkpoint(checkpoint_path=pretrained_checkpoint_path)
    elif pretrained_model_path is not None:
        convo_wizard.from_pretrained(model_path=pretrained_model_path)

    trainer.train_and_eval(**config['train_and_eval']['args'])

    tracker.done()


if __name__ == '__main__':
    parser = ArgumentParser(description='train LM to generate controlled text')
    parser.add_argument('--config_path', type=str, help='path to config file')

    parser.add_argument('--base_path_to_store_results', type=str, help='base path to store results',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--train_data_path', type=str, help='path to the training data', default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--experiment_name', type=str, help='wandb experiment name', default='convo_wizard_experiment')
    parser.add_argument('--project_name', type=str, help='wandb project name', default='convo_wizard')
    parser.add_argument('--entity_name', type=str, help='wandb entity name', default=None)
    parser.add_argument('--log_to_wandb', action='store_true', help='whether to use wandb logging')

    args = parser.parse_args()

    main(config_path=args.config_path, base_path_to_store_results=args.base_path_to_store_results,
         tokenizer_path=args.tokenizer_path, train_data_path=args.train_data_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         experiment_name=args.experiment_name, project_name=args.project_name, entity_name=args.entity_name,
         log_to_wandb=args.log_to_wandb)
