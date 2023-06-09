import os
from argparse import ArgumentParser

import datasets
import torch
import yaml
from torch import nn
from torch.optim import Adam

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.optimizers.noam import NoamOptimizer
from convo_wizard.trainers.trainer import ConvoWizardTrainer
from convo_wizard.utils.tracker import Tracker
from convo_wizard.utils.utils import set_seed


def main(config_path, base_path_to_store_results, tokenizer_path, tokenized_hf_dataset_path, pretrained_model_path=None,
         pretrained_checkpoint_path=None, experiment_name='experiment', project_name='convo_wizard',
         entity_name='cornell-nlp', log_to_wandb=True, resume_wandb_logging=False):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    tracker = Tracker(config=config, base_path_to_store_results=base_path_to_store_results,
                      experiment_name=experiment_name, project_name=project_name, entity_name=entity_name,
                      log_to_wandb=log_to_wandb, resume_wandb_logging=resume_wandb_logging)

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
    tokenized_hf_dataset = datasets.load_from_disk(dataset_path=tokenized_hf_dataset_path)
    tokenized_val_data, tokenized_test_data = None, None
    if 'val' in tokenized_hf_dataset:
        tokenized_val_data = tokenized_hf_dataset['val']

    # Don't use vocab_size: https://github.com/huggingface/transformers/issues/12632#issuecomment-878071592.
    convo_wizard = ConvoWizard(vocab_size=len(convo_uncased_tokenizer),
                               padding_idx=convo_uncased_tokenizer.pad_token_id,
                               cls_or_sep_token_idx=cls_or_sep_token_idx, device=device,
                               **config['transformer']['args'])
    optimizer = NoamOptimizer(Adam(convo_wizard.get_trainable_params(), **config['optimizer']['adam']['args']),
                              embedding_dim=config['transformer']['args']['embedding_dim'],
                              **config['optimizer']['args'])
    trainer = ConvoWizardTrainer(convo_wizard=convo_wizard, optimizer=optimizer, tracker=tracker,
                                 tokenized_train_data=tokenized_hf_dataset['train'],
                                 tokenized_val_data=tokenized_val_data, loss_fn=nn.CrossEntropyLoss, device=device,
                                 **config['trainer']['args']['generator'])

    if pretrained_checkpoint_path is not None:
        trainer.load_from_checkpoint(checkpoint_path=pretrained_checkpoint_path)
    elif pretrained_model_path is not None:
        convo_wizard.from_pretrained(model_path=pretrained_model_path)

    trainer.train_and_eval(**config['train_and_eval']['args']['generator'])

    tracker.done()


if __name__ == '__main__':
    parser = ArgumentParser(description='train LM to generate controlled text')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--base_path_to_store_results', type=str, help='base path to store results',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--tokenized_hf_dataset_path', type=str, help='path to the huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--experiment_name', type=str, help='wandb experiment name', default='convo_wizard_experiment')
    parser.add_argument('--project_name', type=str, help='wandb project name', default='convo_wizard')
    parser.add_argument('--entity_name', type=str, help='wandb entity name', default=None)
    parser.add_argument('--log_to_wandb', action='store_true', help='whether to use wandb logging')
    parser.add_argument('--resume_wandb_logging', action='store_true',
                        help='whether to resume wandb logging from the experiment with the same name')

    args = parser.parse_args()

    main(config_path=args.config_path, base_path_to_store_results=args.base_path_to_store_results,
         tokenizer_path=args.tokenizer_path, tokenized_hf_dataset_path=args.tokenized_hf_dataset_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         experiment_name=args.experiment_name, project_name=args.project_name, entity_name=args.entity_name,
         log_to_wandb=args.log_to_wandb, resume_wandb_logging=args.resume_wandb_logging)
