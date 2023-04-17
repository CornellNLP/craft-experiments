import os
from argparse import ArgumentParser
from pathlib import Path

import datasets
import yaml
from convokit import Corpus, download

from convo_wizard.data_processors.utils import generate_convokit_flat_corpus


def main(config_path, path_to_store_hf_dataset, convokit_download_dir=None, convokit_corpus_dir=None):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    if convokit_corpus_dir is None:
        convokit_download_dir = str(Path.home()) if convokit_download_dir is None else convokit_download_dir
        corpus = Corpus(filename=download(config['convokit']['id'], data_dir=convokit_download_dir))
    else:
        corpus = Corpus(convokit_corpus_dir)
    flat_corpus = generate_convokit_flat_corpus(corpus, **config['convokit']['args'])
    dataset = datasets.Dataset.from_dict({'convos': flat_corpus})
    dataset.to_json(path_to_store_hf_dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='generate training dataset (flat conversational corpus)')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--path_to_store_hf_dataset', type=str, help='path to store huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--convokit_download_dir', type=str, help='path to downloaded convokit corpus',
                        default=str(Path.home()))
    parser.add_argument('--convokit_corpus_dir', type=str, help='path to load the convokit corpus from', default=None)

    args = parser.parse_args()

    main(args.config_path, args.path_to_store_hf_dataset, args.convokit_download_dir, args.convokit_corpus_dir)
