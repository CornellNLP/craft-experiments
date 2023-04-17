import numpy as np


class NoamOptimizer(object):
    """https://arxiv.org/pdf/1706.03762."""

    def __init__(self, optimizer, embedding_dim, num_warmup_steps=4000):
        super().__init__()

        self._optimizer = optimizer
        self._embedding_dim = embedding_dim
        self._num_warmup_steps = num_warmup_steps
        self._step_number = 0
        self.lr = 0

    def __str__(self):
        return f'noam learning rate scheduler \n' \
               f'- optimizer: {self._optimizer} \n' \
               f'- warmup steps: {self._num_warmup_steps} \n' \
               f'- embedding dim: {self._embedding_dim}'

    def state_dict(self):
        return self._optimizer.state_dict()

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def update_lr(self):
        self._step_number = self._step_number + 1

        self.lr = np.power(self._embedding_dim, -0.5) * np.min(
            [np.power(self._step_number, -0.5), self._step_number * np.power(self._num_warmup_steps, -1.5)])
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr
