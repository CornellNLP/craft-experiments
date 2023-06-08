import os
from argparse import ArgumentParser

import datasets
import yaml

from convo_wizard.data_processors.tokenizers import convo_tokenizer, convo_tokenizer_v2
from convo_wizard.data_processors.tokenizers.utils import batch_tokenize


def main(config_path, path_to_store_tokenized_hf_dataset, tokenizer_path, convokit_flat_corpus_hf_filepath,
         split_by_split_col_in_dataset=False, split_train_val_test=False, finetune_as_lm=False):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    tokenize_data_args = config['tokenize_data']['args']['pretrain_or_finetune']
    if finetune_as_lm:
        tokenize_data_args = config['tokenize_data']['args']['finetune_as_lm']

    if tokenize_data_args['train_val']['use_cls']:
        convo_uncased_tokenizer = convo_tokenizer.ConvoTokenizer.load(tokenizer_path)
    else:
        convo_uncased_tokenizer = convo_tokenizer_v2.ConvoTokenizer.load(tokenizer_path)
    dataset = datasets.load_dataset('json', data_files=convokit_flat_corpus_hf_filepath)['train']  # defaults to 'train'

    train_data, val_data, test_data = None, None, None
    tokenized_dataset_dict = {}
    tokenize_train_val_helper = lambda data_instance: batch_tokenize(data_instance,
                                                                     pretrained_tokenizer=convo_uncased_tokenizer,
                                                                     **tokenize_data_args['train_val'])
    tokenize_val_test_helper = tokenize_train_val_helper
    if 'label' in dataset.features:
        tokenize_val_test_helper = lambda data_instance: batch_tokenize(data_instance,
                                                                        pretrained_tokenizer=convo_uncased_tokenizer,
                                                                        **tokenize_data_args['val_test'])
    if split_by_split_col_in_dataset:
        train_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['train'])
        val_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['val'])
        test_data = dataset.filter(lambda convo: convo['split'] == config['split_cols_in_dataset']['test'])
    elif split_train_val_test:
        train_rest = dataset.train_test_split(test_size=(config['splits']['val'] + config['splits']['test']))
        if config['splits']['test'] != 0:
            train_data = train_rest['train']
            val_test = train_rest['test'].train_test_split(test_size=config['splits']['test'])
            val_data, test_data = val_test['train'], val_test['test']
        else:
            train_data = train_rest['train']
            val_data = train_rest['test']

    # https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size
    keep_cols = {'input_ids', 'position_ids', 'attention_mask', 'token_type_ids', 'relative_position_ids', 'labels'}
    if tokenize_data_args['train_val']['use_cls']:
        keep_cols.add('cls_mask')
        mask_to_remove = 'sep_mask'
    else:
        keep_cols.add('sep_mask')
        mask_to_remove = 'cls_mask'
    train_remove_cols = list(set(train_data.column_names) - keep_cols)
    val_remove_cols = list(set(val_data.column_names) - keep_cols)

    tokenized_dataset_dict['train'] = train_data.map(tokenize_train_val_helper, batched=True,
                                                     remove_columns=train_remove_cols)
    tokenized_dataset_dict['val'] = val_data.map(tokenize_train_val_helper, batched=True,
                                                 remove_columns=val_remove_cols)
    if 'label' in dataset.features:
        if finetune_as_lm:
            tokenized_dataset_dict['val_unpadded'] = val_data.map(tokenize_val_test_helper, batched=True,
                                                                  remove_columns=val_remove_cols)
        else:
            tokenized_dataset_dict['val_each_utt_label'] = val_data.map(tokenize_val_test_helper, batched=True,
                                                                        remove_columns=val_remove_cols)
    if test_data is not None:
        test_remove_cols = list(set(test_data.column_names) - keep_cols)
        tokenized_dataset_dict['test'] = test_data.map(tokenize_val_test_helper, batched=True,
                                                       remove_columns=test_remove_cols)
    tokenized_data = datasets.DatasetDict(tokenized_dataset_dict)
    tokenized_data = tokenized_data.remove_columns(mask_to_remove)  # remove the unwanted mask
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
    parser.add_argument('--finetune_as_lm', action='store_true', help='whether to populate classification data as lm')

    args = parser.parse_args()

    main(config_path=args.config_path, path_to_store_tokenized_hf_dataset=args.path_to_store_tokenized_hf_dataset,
         tokenizer_path=args.tokenizer_path, convokit_flat_corpus_hf_filepath=args.convokit_flat_corpus_hf_filepath,
         split_by_split_col_in_dataset=args.split_by_split_col_in_dataset,
         split_train_val_test=args.split_train_val_test, finetune_as_lm=args.finetune_as_lm)
