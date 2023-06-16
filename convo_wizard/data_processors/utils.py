import re

import matplotlib as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


def reddit_text_processor(text, tokenizer):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    if tokenizer is not None:
        return [tok for tok in tokenizer(text) if len(tok) > 1]
    return text


def generate_convokit_flat_corpus(corpus, text_processor=None, min_num_comments=1, path_len=(-1, -1),
                                  max_num_convos=None, plot_stats=True, label_col_name=None, split_col_name=None):
    def _stats_plotter(distr, xlabel, ylabel='frequency', color='tab:blue'):
        plt.hist(distr, alpha=0.5, color=color)

        avg_distr = np.mean(distr)
        min_ylim, max_ylim = plt.ylim()
        plt.axvline(avg_distr, color=color)
        plt.text(avg_distr * 1.04, max_ylim * 0.9, 'mean: {:.2f}'.format(avg_distr))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    flat_dataset, labels, splits = [], [], []
    num_utts_per_conv, utt_lengths = [], []
    if text_processor is None:
        text_processor = reddit_text_processor

    if max_num_convos is None or max_num_convos > len(corpus.conversations):
        max_num_convos = len(corpus.conversations)

    path_len = list(path_len)
    if path_len[0] == -1:
        path_len[0] = 2
    if path_len[1] == -1:
        path_len[1] = np.inf

    for conv_idx, (conv_id, conv) in tqdm(enumerate(corpus.conversations.items())):
        if conv_idx >= max_num_convos:
            break

        # The 'conversations-got-awry-cmv-corpus' doesn't have num_comments in the 'meta' field.
        if ('num_comments' in conv.meta and conv.meta['num_comments'] >= min_num_comments) or (
                'num_comments' not in conv.meta):
            try:
                for path in conv.get_root_to_leaf_paths():
                    if path_len[0] <= len(path) <= path_len[1]:
                        convo_utts = [text_processor(utt.text, tokenizer=None) for utt in path]
                        flat_dataset.append(convo_utts)
                        if label_col_name is not None and label_col_name in conv.meta:
                            labels.append(conv.meta[label_col_name])
                        if split_col_name is not None and split_col_name in conv.meta:
                            splits.append(conv.meta[split_col_name])
                        if plot_stats:
                            num_utts_per_conv.append(len(path))
                            utt_lengths = utt_lengths + [len(utt_text.split()) for utt_text in convo_utts]
            except:
                pass

    if plot_stats:
        _stats_plotter(num_utts_per_conv, xlabel='num utts per conv', color='tab:blue')
        _stats_plotter(utt_lengths, xlabel='utt space-sep token length', color='tab:orange')

    return flat_dataset, labels, splits


def get_torch_dataset(tokenized_dataset, is_labeled_data=False):
    cls_or_sep_mask_colname = 'sep_mask' if 'sep_mask' in set(tokenized_dataset.column_names) else 'cls_mask'
    if is_labeled_data:
        tokenized_dataset.set_format(type='torch',
                                     columns=(['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                               'attention_mask', 'labels'] + [cls_or_sep_mask_colname]))
    else:
        tokenized_dataset.set_format(type='torch',
                                     columns=(['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                               'attention_mask'] + [cls_or_sep_mask_colname]))

    return tokenized_dataset


def get_dataloader(torch_dataset, batch_size=128, shuffle=False, num_samples=None, num_workers=0):
    if num_samples is not None and shuffle is True:
        replacement = False
        if num_samples > len(torch_dataset):
            replacement = True
        sampler = RandomSampler(torch_dataset, replacement=replacement, num_samples=num_samples)
        return DataLoader(torch_dataset, sampler=sampler, batch_size=batch_size, pin_memory=True,
                          num_workers=num_workers)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
