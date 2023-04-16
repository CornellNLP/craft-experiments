from itertools import chain

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from torch import nn
from tqdm import tqdm, trange

from convo_wizard.data_processors.utils import get_dataloader
from convo_wizard.utils.utils import device_mapper


class ConvoWizardTrainer(nn.Module):
    def __init__(self, convo_wizard, optimizer, tokenized_train_data, tokenized_val_data, tracker=None,
                 is_labeled_data=False, use_relative_position_ids=False, loss_fn=nn.CrossEntropyLoss,
                 labels_ignore_idx=0, use_class_weights=False, batch_size=64, gradient_clip_value=None,
                 device=torch.device('cpu')):
        super().__init__()

        self._device = device
        self._model = convo_wizard
        self._use_relative_position_ids = use_relative_position_ids
        self._is_labeled_data = is_labeled_data
        self._optimizer = optimizer
        self._gradient_clip_value = gradient_clip_value
        self._batch_size = batch_size
        self._tracker = tracker

        self._train_dataloader = get_dataloader(tokenized_train_data, batch_size=batch_size,
                                                is_labeled_data=is_labeled_data, shuffle=True)
        self._val_dataloader = None
        if tokenized_val_data is not None:
            self._val_dataloader = get_dataloader(tokenized_val_data, batch_size=batch_size,
                                                  is_labeled_data=is_labeled_data, shuffle=False)

        self._labels_ignore_idx = labels_ignore_idx
        class_weights = None
        if use_class_weights and is_labeled_data:
            class_weights = self._compute_class_weights(tokenized_train_data, tokenized_val_data)
            print(f'class weights: {class_weights}')
        self._loss_fn = loss_fn(ignore_index=labels_ignore_idx, weight=class_weights)

    def _compute_class_weights(self, tokenized_train_data, tokenized_val_data):
        train_labels = list(chain.from_iterable(tokenized_train_data['labels'].tolist()))
        train_labels = [_ for _ in train_labels if _ != self._labels_ignore_idx]
        val_labels = []
        if self._val_dataloader is not None:
            val_labels = list(chain.from_iterable(tokenized_val_data['labels'].tolist()))
            val_labels = [_ for _ in val_labels if _ != self._labels_ignore_idx]

        all_labels = train_labels + val_labels
        label_ids = np.unique(all_labels)

        class_weights = class_weight.compute_class_weight('balanced', classes=label_ids, y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def _compute_loss(self, predictions, labels):
        predictions = device_mapper(predictions, torch.device('cpu'))
        labels = device_mapper(labels, torch.device('cpu'))

        # predictions: (batch_size, max_length, output_dim)
        # labels: (batch_size * max_length)
        return self._loss_fn(predictions, labels)

    def _compute_metrics(self, predictions, labels, ce_loss):
        if not self._is_labeled_data:
            metrics = {'perplexity': np.exp(ce_loss)}
        else:
            # predictions: (batch_size * max_length, output_dim)
            # labels: (batch_size * max_length)
            # max_predictions: (batch_size * max_length, 1)
            max_predictions = predictions.argmax(dim=-1)
            mask = (labels != self._labels_ignore_idx).nonzero()

            y_true, y_pred = labels[mask].numpy(), max_predictions[mask].numpy()
            metrics = {'precision': precision_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'recall': recall_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'f1': f1_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred)}
        return metrics

    def _train_epoch(self, dataloader):
        batch_losses, epoch_loss = [], 0.0
        if not self._is_labeled_data:
            all_batches_metrics = {'perplexity': []}
        else:
            all_batches_metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
        epoch_metrics = {}

        self._model.train()
        for data_batch in tqdm(dataloader):
            self._optimizer.zero_grad()

            if self._use_relative_position_ids:
                position_ids = data_batch['relative_position_ids']
            else:
                position_ids = data_batch['position_ids']
            # lm_output = (batch_size, max_length, vocab_size)
            # classifier_output = (batch_size, max_length, output_dim)
            lm_output, classifier_output = self._model(input_ids=data_batch['input_ids'], position_ids=position_ids,
                                                       token_type_ids=data_batch['token_type_ids'],
                                                       attention_mask=data_batch['attention_mask'],
                                                       make_predictions=self._is_labeled_data)
            if self._is_labeled_data:
                # predictions: (batch_size * max_length, output_dim)
                predictions = classifier_output.view(-1, lm_output.shape[-1])
                # data_batch['labels']: (batch_size, max_length)
                # labels: (batch_size * max_length)
                labels = data_batch['labels'].view(-1)
            else:
                # predictions = (batch_size * max_length, vocab_size)
                predictions = lm_output.view(-1, lm_output.shape[-1])
                # position_ids: (batch_size, max_length)
                # labels: (batch_size * max_length)
                labels = position_ids.view(-1)

            batch_loss = self._compute_loss(predictions=predictions, labels=labels)
            batch_loss.backward()
            batch_loss = batch_loss.item()
            if self._gradient_clip_value is not None:
                nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_value)
            self._optimizer.step()
            self._optimizer.update_lr()

            batch_metrics = self._compute_metrics(predictions, labels, ce_loss=batch_loss)
            for metric in all_batches_metrics.keys():
                all_batches_metrics[metric].append(batch_metrics[metric])
            batch_losses.append(batch_loss)

        epoch_metrics['loss'] = sum(batch_losses) / len(dataloader)
        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def _eval_epoch(self, dataloader):
        batch_losses, epoch_loss = [], 0.0
        if not self._is_labeled_data:
            all_batches_metrics = {'perplexity': []}
        else:
            all_batches_metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
        epoch_metrics = {}

        self._model.eval()
        with torch.no_grad:
            for data_batch in tqdm(dataloader):
                if self._use_relative_position_ids:
                    position_ids = data_batch['relative_position_ids']
                else:
                    position_ids = data_batch['position_ids']
                # lm_output = (batch_size, max_length, vocab_size)
                # classifier_output = (batch_size, max_length, output_dim)
                lm_output, classifier_output = self._model(input_ids=data_batch['input_ids'], position_ids=position_ids,
                                                           token_type_ids=data_batch['token_type_ids'],
                                                           attention_mask=data_batch['attention_mask'],
                                                           make_predictions=self._is_labeled_data)
            if self._is_labeled_data:
                # predictions: (batch_size * max_length, output_dim)
                predictions = classifier_output.view(-1, lm_output.shape[-1])
                # data_batch['labels']: (batch_size, max_length)
                # labels: (batch_size * max_length)
                labels = data_batch['labels'].view(-1)
            else:
                # predictions = (batch_size * max_length, vocab_size)
                predictions = lm_output.view(-1, lm_output.shape[-1])
                # position_ids: (batch_size, max_length)
                # labels: (batch_size * max_length)
                labels = position_ids.view(-1)

            batch_loss = self._compute_loss(predictions=predictions, labels=labels).item()
            batch_metrics = self._compute_metrics(predictions, labels, ce_loss=batch_loss)
            for metric in all_batches_metrics.keys():
                all_batches_metrics[metric].append(batch_metrics[metric])
            batch_losses.append(batch_loss)

        epoch_metrics['val_loss'] = sum(batch_losses) / len(dataloader)
        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def train_and_eval(self, num_epochs):
        for epoch in trange(num_epochs):
            train_metrics = self._train_epoch(self._train_dataloader)
            val_metrics = None
            if self._val_dataloader is not None:
                val_metrics = self._eval_epoch(self._val_dataloader)

            if self._tracker is not None:
                self._tracker.log_metrics(epoch=epoch, split_name='train', metrics=train_metrics)
                if self._val_dataloader is not None:
                    self._tracker.log_metrics(epoch=epoch, split_name='val', metrics=val_metrics)
                self._tracker.save_auto_model(self._model)

    @staticmethod
    @torch.no_grad()
    def generate(self, max_new_tokens, top_k=None):
        pass

    @staticmethod
    @torch.no_grad()
    def predict(self, tokenized_test_data, batch_size=128):
        pass
