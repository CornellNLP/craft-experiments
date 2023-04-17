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
                                  max_num_convos=None, plot_stats=True):
    def _stats_plotter(distr, xlabel, ylabel='frequency', color='tab:blue'):
        plt.hist(distr, alpha=0.5, color=color)

        avg_distr = np.mean(distr)
        min_ylim, max_ylim = plt.ylim()
        plt.axvline(avg_distr, color=color)
        plt.text(avg_distr * 1.04, max_ylim * 0.9, 'mean: {:.2f}'.format(avg_distr))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    flat_dataset = []
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

        if conv.meta['num_comments'] >= min_num_comments:
            try:
                for path in conv.get_root_to_leaf_paths():
                    if path_len[0] <= len(path) <= path_len[1]:
                        convo_utts = [text_processor(utt.text, tokenizer=None) for utt in path]
                        flat_dataset.append(convo_utts)
                        if plot_stats:
                            num_utts_per_conv.append(len(path))
                            utt_lengths = utt_lengths + [len(utt_text.split()) for utt_text in convo_utts]
            except:
                pass

    if plot_stats:
        _stats_plotter(num_utts_per_conv, xlabel='num utts per conv', color='tab:blue')
        _stats_plotter(utt_lengths, xlabel='utt space-sep token length', color='tab:orange')

    return flat_dataset


def get_torch_dataset(tokenized_dataset, is_labeled_data=False):
    if is_labeled_data:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                              'attention_mask', 'labels'])
    else:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                              'attention_mask'])

    return tokenized_dataset


def get_dataloader(torch_dataset, batch_size=128, shuffle=False, num_samples=None):
    if num_samples is not None and shuffle is True:
        sampler = RandomSampler(torch_dataset, replacement=False, num_samples=num_samples)
        return DataLoader(torch_dataset, sampler=sampler, batch_size=batch_size)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
