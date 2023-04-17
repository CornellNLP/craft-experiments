import os
from argparse import ArgumentParser

import datasets
import yaml

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer
from convo_wizard.data_processors.tokenizers.utils import batch_tokenize


def main(config_path, path_to_store_tokenized_hf_dataset, tokenizer_path, convokit_flat_corpus_hf_filepath):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    convo_uncased_tokenizer = ConvoTokenizer.load(tokenizer_path)
    dataset = datasets.load_dataset('json', data_files={'train': convokit_flat_corpus_hf_filepath})

    tokenize_helper = lambda data_instance: batch_tokenize(data_instance, pretrained_tokenizer=convo_uncased_tokenizer,
                                                           **config['tokenize_data']['args'])
    tokenized_data = dataset.map(tokenize_helper, batched=True)['train']
    tokenized_data.to_json(path_to_store_tokenized_hf_dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='generate training dataset (flat conversational corpus)')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--path_to_store_tokenized_hf_dataset', type=str, help='path to store huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to load the tokenizer from', default=None)
    parser.add_argument('--convokit_flat_corpus_hf_filepath', type=str, help='path to load the convokit corpus from',
                        default=None)

    args = parser.parse_args()

    main(args.config_path, args.path_to_store_tokenized_hf_dataset, args.tokenizer_path,
         args.convokit_flat_corpus_hf_filepath)
