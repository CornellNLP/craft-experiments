import os
from argparse import ArgumentParser

import datasets
import yaml

from convo_wizard.data_processors.tokenizers.convo_tokenizer import ConvoTokenizer


def main(config_path, convokit_flat_corpus_hf_filepath, path_to_store_hf_tokenizer):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    hf_dataset = datasets.load_dataset('json', data_files={'train': convokit_flat_corpus_hf_filepath})
    tokenizer = ConvoTokenizer(hf_dataset, **config['tokenizer']['args'])
    tokenizer.save(path_to_store_hf_tokenizer)


if __name__ == '__main__':
    parser = ArgumentParser(description='generate training dataset (flat conversational corpus)')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--convokit_flat_corpus_hf_filepath', type=str, help='path to the huggingface training dataset',
                        default=os.getcwd())
    parser.add_argument('--path_to_store_hf_tokenizer', type=str, help='path to store the tokenizer',
                        default=os.getcwd())

    args = parser.parse_args()

    main(config_path=args.config_path, convokit_flat_corpus_hf_filepath=args.convokit_flat_corpus_hf_filepath,
         path_to_store_hf_tokenizer=args.path_to_store_hf_tokenizer)
