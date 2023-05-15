import os
from argparse import ArgumentParser

import datasets
import yaml

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer
from convo_wizard.data_processors.tokenizers.utils import batch_tokenize


def main(config_path, path_to_store_tokenized_hf_dataset, tokenizer_path, convokit_flat_corpus_hf_filepath,
         split_by_split_col_in_dataset=False, split_train_val_test=False):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    convo_uncased_tokenizer = ConvoTokenizer.load(tokenizer_path)
    dataset = datasets.load_dataset('json', data_files=convokit_flat_corpus_hf_filepath)['train']  # defaults to 'train'
    if split_by_split_col_in_dataset:
        train_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['train'])
        val_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['val'])
        test_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['test'])
        dataset = \
            datasets.DatasetDict({'train': train_data, 'val': val_data, 'test': test_data})
    elif split_train_val_test:
        train_rest = dataset.train_test_split(test_size=(config['splits']['val'] + config['splits']['test']))
        val_test = train_rest['test'].train_test_split(test_size=config['splits']['test'])
        dataset = \
            datasets.DatasetDict({'train': train_rest['train'], 'val': val_test['train'], 'test': val_test['test']})

    tokenize_helper = lambda data_instance: batch_tokenize(data_instance, pretrained_tokenizer=convo_uncased_tokenizer,
                                                           **config['tokenize_data']['args'])
    tokenized_data = dataset.map(tokenize_helper, batched=True)
    tokenized_data.save_to_disk(path_to_store_tokenized_hf_dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='generate training dataset (flat conversational corpus)')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--path_to_store_tokenized_hf_dataset', type=str, help='path to store huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to load the tokenizer from', default=None)
    parser.add_argument('--convokit_flat_corpus_hf_filepath', type=str, help='path to load the convokit corpus from',
                        default=None)
    parser.add_argument('--split_by_split_col_in_dataset', action='store_true',
                        help='whether to split the dataset using the "split" column in the dataset')
    parser.add_argument('--split_train_val_test', action='store_true', help='whether to split the dataset')

    args = parser.parse_args()

    main(config_path=args.config_path, path_to_store_tokenized_hf_dataset=args.path_to_store_tokenized_hf_dataset,
         tokenizer_path=args.tokenizer_path, convokit_flat_corpus_hf_filepath=args.convokit_flat_corpus_hf_filepath,
         split_by_split_col_in_dataset=args.split_by_split_col_in_dataset,
         split_train_val_test=args.split_train_val_test)
