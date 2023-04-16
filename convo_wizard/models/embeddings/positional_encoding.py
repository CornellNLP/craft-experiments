import matplotlib as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

from convo_wizard.utils.utils import device_mapper


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, pad_token_position=0, max_length=2048, freq_base=10000,
                 device=torch.device('cpu')):
        super().__init__()

        self._device = device
        self._pad_token_position = pad_token_position  # used for [PAD] identification
        self._num_positions = max_length  # excludes [PAD] position
        self._freq_base = freq_base

        # One position reserved for padding tokens.
        num_positions = max_length + 1
        self._position_encoding = nn.Embedding(num_embeddings=num_positions, embedding_dim=embedding_dim,
                                               padding_idx=self._pad_token_position, device=self._device)
        self._position_encoding.weight.data = \
            self.sinusoidal_weight_init(num_positions=num_positions, embedding_dim=embedding_dim,
                                        freq_base=self._freq_base, pad_token_position=self._pad_token_position,
                                        device=self._device)

        # Freeze the position encodings (no training needed).
        self._position_encoding.weight.requires_grad = False

    @property
    def type(self):
        return 'positional_encoding'

    @staticmethod
    def sinusoidal_weight_init(num_positions, embedding_dim, freq_base=10000, pad_token_position=0,
                               device=torch.device('cpu')):
        positional_encodings = []
        for position in range(num_positions):
            if position == pad_token_position:
                positional_encoding = np.zeros(embedding_dim)
            else:
                positional_encoding = [position / np.power(freq_base, 2 * ((k // 2) + 1) / embedding_dim)
                                       for k in range(0, embedding_dim, 1)]
            positional_encodings.append(np.array(positional_encoding))
        positional_encodings = np.array(positional_encodings)

        positional_encodings[1:, 0::2] = np.sin(positional_encodings[1:, 0::2])
        positional_encodings[1:, 1::2] = np.cos(positional_encodings[1:, 1::2])

        return torch.tensor(positional_encodings, device=device).float()

    def forward(self, position_ids):
        # position_ids: (batch_size, max_length)
        position_ids = device_mapper(position_ids, self._device)

        # position_encodings: (batch_size, max_length, embedding_dim)
        position_encodings = self._position_encoding(position_ids)
        assert position_encodings.requires_grad is False

        return position_encodings

    def plot_encodings(self):
        position_encodings = self._position_encoding.weight.T.numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        sns.heatmap(position_encodings, vmin=position_encodings.min(),
                    vmax=position_encodings.max(), ax=ax, cbar=True)
        ax.set_xlabel('position in the sequence')
        ax.set_ylabel('encoding')

        return plt

    def plot_samples(self, plot_every_n=5, num_positions=21):
        position_encodings = self._position_encoding.weight.T.numpy()

        fig, axs = plt.subplots(2, 2, figsize=(10, 3))
        axs = [_ for ax in axs for _ in ax]
        for i in range(len(axs)):
            axs[i].plot(np.arange(0, num_positions), position_encodings[i * plot_every_n, : num_positions],
                        color=f'C{i}', marker="o", markersize=6, markeredgecolor='black')
            axs[i].set_title(f'positional-{i * plot_every_n}')
            axs[i].set_xlabel('position in sequence')
            axs[i].set_ylabel('encoding')
            axs[i].set_xticks(np.arange(0, num_positions))
            axs[i].set_ylim(-1.2, 1.2)
        fig.subplots_adjust(hspace=0.8)

        return plt
