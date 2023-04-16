import os
from argparse import ArgumentParser

import datasets
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torchinfo import summary

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.optimizers.noam import NoamOptimizer
from convo_wizard.trainers.trainer import ConvoWizardTrainer
from convo_wizard.utils.tracker import Tracker


def main(config_path, base_path_to_store_results, tokenizer_path, train_data_path, experiment_name,
         project_name='convo_wizard', entity_name='cornell-nlp', log_to_wandb=True):
    # TODO: integrate the .yaml file to load the config.
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    tracker = Tracker(config=config, base_path_to_store_results=base_path_to_store_results,
                      experiment_name=experiment_name, project_name=project_name, entity_name=entity_name,
                      log_to_wandb=log_to_wandb)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    convo_uncased_tokenizer = ConvoTokenizer.load(tokenizer_path)
    tokenized_train_data = datasets.load_dataset('json', data_files={'train': train_data_path})['train']

    convo_wizard = ConvoWizard(vocab_size=convo_uncased_tokenizer.vocab_size, embedding_dim=300, hidden_dim=512,
                               max_relative_position=None, use_sinusoidal_init=True, positional_network_type='ffnn',
                               output_dim=2, classifier_head_type='rnn', num_heads=6, num_encoder_layers=4,
                               padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_token_idx=convo_uncased_tokenizer.cls_token_id, labels_ignore_idx=-100,
                               max_length=2048, pad_token_position=0, pad_tok_type=0, num_token_types=2,
                               attention_dropout=0.05, dropout=0.1, device=device)
    optimizer = NoamOptimizer(Adam(convo_wizard.get_trainable_params(), lr=2e-3, betas=(0.9, 0.999), eps=1e-9),
                              embedding_dim=300, num_warmup_steps=4000)
    trainer = ConvoWizardTrainer(convo_wizard=convo_wizard, optimizer=optimizer, tracker=tracker,
                                 tokenized_train_data=tokenized_train_data, tokenized_val_data=None,
                                 is_labeled_data=False, loss_fn=nn.CrossEntropyLoss, labels_ignore_idx=0,
                                 use_class_weights=False, batch_size=1, gradient_clip_value=0.75, device=device)
    summary(convo_wizard)

    trainer.train_and_eval(num_epochs=1)


if __name__ == '__main__':
    parser = ArgumentParser(description='train LM to generate controlled text')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--project_name', type=str, help='wandb project name', default='convo_wizard')
    parser.add_argument('--experiment_name', type=str, help='wandb experiment name', default='convo_wizard_experiment')
    parser.add_argument('--entity_name', type=str, help='wandb entity name', default=None)
    parser.add_argument('--base_path_to_store_results', type=str, help='base path to store results',
                        default=os.getcwd())
    parser.add_argument('--log_to_wandb', action='store_true', help='whether to use wandb logging')

    # TODO: remove the following args and use config file instead.
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--train_data_path', type=str, help='path to the training data', default=os.getcwd())

    args = parser.parse_args()

    main(args.config_path, args.base_path_to_store_results, args.tokenizer_path, args.train_data_path,
         args.experiment_name, args.project_name, args.entity_name, args.log_to_wandb)
