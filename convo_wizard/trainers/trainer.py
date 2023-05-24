from itertools import chain

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm, trange

from convo_wizard.data_processors.utils import get_torch_dataset, get_dataloader
from convo_wizard.utils.utils import device_mapper


class ConvoWizardTrainer(nn.Module):
    def __init__(self, convo_wizard, optimizer, tokenized_train_data, tokenized_val_data, tracker=None,
                 is_labeled_data=False, use_relative_position_ids=False, loss_fn=nn.CrossEntropyLoss,
                 labels_ignore_idx=0, use_class_weights=False, gradient_clip_value=None, num_workers=0,
                 use_mixed_precision=True, device=torch.device('cpu')):
        super().__init__()

        self._device = device
        self._model = convo_wizard
        self._use_relative_position_ids = use_relative_position_ids
        self._is_labeled_data = is_labeled_data
        self._optimizer = optimizer
        self._gradient_clip_value = gradient_clip_value
        self._tracker = tracker

        self._use_mixed_precision = use_mixed_precision
        if self._device.type != 'cuda':
            # https://discuss.pytorch.org/t/error-while-using-16-bit-floats-half/139465/2
            self._use_mixed_precision = False
        self._grad_scaler = GradScaler(enabled=self._use_mixed_precision)

        self._num_workers = num_workers
        self._tokenized_train_data = get_torch_dataset(tokenized_train_data, is_labeled_data=self._is_labeled_data)
        self._tokenized_val_data = None
        if tokenized_val_data is not None:
            self._tokenized_val_data = get_torch_dataset(tokenized_val_data, is_labeled_data=self._is_labeled_data)

        self._labels_ignore_idx = labels_ignore_idx
        class_weights = None
        if use_class_weights and is_labeled_data:
            class_weights = self._compute_class_weights(tokenized_train_data, tokenized_val_data)
            print(f'class weights: {class_weights}')
        self._loss_fn = loss_fn(ignore_index=labels_ignore_idx, weight=class_weights)

        # Class variable to store the starting epoch, especially if loading pretrained model.
        self._start_epoch = 0

    def save_checkpoint(self, epoch, checkpoint_path):
        torch.save({'epoch': epoch, 'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'scaler_state_dict': self._grad_scaler.state_dict()}, checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self._start_epoch = checkpoint['epoch'] + 1

    def _compute_class_weights(self, tokenized_train_data, tokenized_val_data):
        train_labels = list(chain.from_iterable(tokenized_train_data['labels'].tolist()))
        train_labels = [_ for _ in train_labels if _ != self._labels_ignore_idx]
        val_labels = []
        if self._tokenized_val_data is not None:
            val_labels = list(chain.from_iterable(tokenized_val_data['labels'].tolist()))
            val_labels = [_ for _ in val_labels if _ != self._labels_ignore_idx]

        all_labels = train_labels + val_labels
        label_ids = np.unique(all_labels)

        class_weights = class_weight.compute_class_weight('balanced', classes=label_ids, y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=self._device)
        return class_weights

    def _compute_loss(self, predictions, labels):
        predictions = device_mapper(predictions, self._device)
        labels = device_mapper(labels, self._device)

        # predictions: (batch_size, max_length, output_dim)
        # labels: (batch_size * max_length)
        return self._loss_fn(predictions, labels)

    def _compute_metrics(self, predictions, labels, ce_loss):
        if not self._is_labeled_data:
            metrics = {'loss': ce_loss, 'perplexity': np.exp(ce_loss)}
        else:
            # predictions: (batch_size * max_length, output_dim)
            # labels: (batch_size * max_length)
            # max_predictions: (batch_size * max_length)
            max_predictions = predictions.argmax(dim=-1)
            mask = (labels != self._labels_ignore_idx).nonzero()

            y_true, y_pred = labels[mask].cpu().numpy(), max_predictions[mask].cpu().numpy()
            metrics = {'loss': ce_loss,
                       'precision': precision_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'recall': recall_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'f1': f1_score(y_true=y_true, y_pred=y_pred, zero_division=0),
                       'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred)}
        return metrics

    def _train_epoch(self, dataloader):
        if not self._is_labeled_data:
            all_batches_metrics = {'loss': [], 'perplexity': []}
        else:
            all_batches_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
        epoch_metrics = {}

        self._model.train()
        for data_batch in tqdm(dataloader):
            # https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide
            self._optimizer.zero_grad(set_to_none=True)

            if self._use_relative_position_ids:
                position_ids = data_batch['relative_position_ids']
            else:
                position_ids = data_batch['position_ids']

            # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            with autocast(device_type=self._device.type, dtype=torch.float16, enabled=self._use_mixed_precision):
                # lm_output = (batch_size, max_length, vocab_size)
                # classifier_output = (batch_size, max_length, output_dim)
                lm_output, classifier_output = self._model(input_ids=data_batch['input_ids'], position_ids=position_ids,
                                                           token_type_ids=data_batch['token_type_ids'],
                                                           attention_mask=data_batch['attention_mask'],
                                                           make_predictions=self._is_labeled_data)
                if self._is_labeled_data:
                    # predictions: (batch_size * max_length, output_dim)
                    predictions = classifier_output.view(-1, classifier_output.shape[-1])
                    # data_batch['labels']: (batch_size, max_length)
                    # labels: (batch_size * max_length)
                    labels = data_batch['labels'].view(-1)
                else:
                    # predictions = (batch_size * (max_length - 1), vocab_size)
                    predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1])
                    # input_ids: (batch_size, max_length)
                    # labels: (batch_size * (max_length - 1))
                    # Note: using circ shift makes the transformer learn the "shift" operator if the attention mask
                    # isn't populated equipped to "not" look at the next tokens! (See "causal" argument in the
                    # multi-head attention module.)
                    #   labels (ignore last circ shift): (batch_size * (max_length - 1))
                    #   labels = torch.roll(data_batch['input_ids'], shifts=-1, dims=1)[:, :-1].contiguous().view(-1)
                    labels = data_batch['input_ids'][:, 1:].contiguous().view(-1)

                batch_loss = self._compute_loss(predictions=predictions, labels=labels)

            self._grad_scaler.scale(batch_loss).backward()
            batch_loss = batch_loss.item()
            self._grad_scaler.unscale_(self._optimizer)
            if self._gradient_clip_value is not None:
                nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_value)
            try:
                self._optimizer.step(grad_scaler=self._grad_scaler)
                self._optimizer.update_lr()
            except TypeError:
                self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()

            batch_metrics = self._compute_metrics(predictions, labels, ce_loss=batch_loss)
            for metric in all_batches_metrics.keys():
                all_batches_metrics[metric].append(batch_metrics[metric])

        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def _eval_epoch(self, dataloader):
        if not self._is_labeled_data:
            all_batches_metrics = {'loss': [], 'perplexity': []}
        else:
            all_batches_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
        epoch_metrics = {}

        self._model.eval()
        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                if self._use_relative_position_ids:
                    position_ids = data_batch['relative_position_ids']
                else:
                    position_ids = data_batch['position_ids']

                # https://discuss.pytorch.org/t/mixed-precision-for-validation/92319/2
                with autocast(device_type=self._device.type, dtype=torch.float16, enabled=self._use_mixed_precision):
                    # lm_output = (batch_size, max_length, vocab_size)
                    # classifier_output = (batch_size, max_length, output_dim)
                    lm_output, classifier_output = self._model(input_ids=data_batch['input_ids'],
                                                               position_ids=position_ids,
                                                               token_type_ids=data_batch['token_type_ids'],
                                                               attention_mask=data_batch['attention_mask'],
                                                               make_predictions=self._is_labeled_data)
                    if self._is_labeled_data:
                        # predictions: (batch_size * max_length, output_dim)
                        predictions = classifier_output.view(-1, classifier_output.shape[-1])
                        # data_batch['labels']: (batch_size, max_length)
                        # labels: (batch_size * max_length)
                        labels = data_batch['labels'].view(-1)
                    else:
                        # predictions = (batch_size * (max_length - 1), vocab_size)
                        predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1])
                        # input_ids: (batch_size, max_length)
                        # labels: (batch_size * (max_length - 1))
                        labels = data_batch['input_ids'][:, 1:].contiguous().view(-1)

                    batch_loss = self._compute_loss(predictions=predictions, labels=labels).item()

                batch_metrics = self._compute_metrics(predictions, labels, ce_loss=batch_loss)
                for metric in all_batches_metrics.keys():
                    all_batches_metrics[metric].append(batch_metrics[metric])

        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def train_and_eval(self, batch_size=64, num_steps_per_epoch=None, num_epochs=8, checkpoint_every=10):
        val_dataloader = None
        if self._tokenized_val_data is not None:
            val_dataloader = get_dataloader(self._tokenized_val_data, batch_size=batch_size, shuffle=False,
                                            num_samples=num_steps_per_epoch, num_workers=self._num_workers)

        for epoch in trange(self._start_epoch, self._start_epoch + num_epochs, 1):
            train_dataloader = get_dataloader(self._tokenized_train_data, batch_size=batch_size, shuffle=True,
                                              num_samples=num_steps_per_epoch, num_workers=self._num_workers)

            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = self._eval_epoch(val_dataloader) if val_dataloader is not None else None

            if self._tracker is not None:
                self._tracker.log_metrics(epoch=epoch, split_name='train', metrics=train_metrics)
                if val_metrics is not None:
                    self._tracker.log_metrics(epoch=epoch, split_name='val', metrics=val_metrics)
                if (epoch + 1) % checkpoint_every == 0:
                    self._tracker.save_checkpoint(self, epoch=epoch)

        self._tracker.save_model(self._model)

    @staticmethod
    @torch.no_grad()
    def test(convo_wizard, tokenized_test_data, prediction_threshold=0.5, batch_size=128, labels_ignore_idx=-100,
             use_relative_position_ids=False, use_mixed_precision=True, num_workers=0, tracker=None,
             device=torch.device('cpu')):
        test_dataloader = get_dataloader(get_torch_dataset(tokenized_test_data, is_labeled_data=True),
                                         batch_size=batch_size, shuffle=False, num_workers=num_workers)

        preds = {'lm': {'y_true': [], 'y_pred': []}, 'cls': {'y_true': [], 'y_pred': []}}

        convo_wizard.eval()
        with torch.no_grad():
            for data_batch in tqdm(test_dataloader):
                if use_relative_position_ids:
                    position_ids = data_batch['relative_position_ids']
                else:
                    position_ids = data_batch['position_ids']

                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                    # classifier_output = (batch_size, max_length, output_dim)
                    _, classifier_output = convo_wizard(input_ids=data_batch['input_ids'],
                                                        position_ids=position_ids,
                                                        token_type_ids=data_batch['token_type_ids'],
                                                        attention_mask=data_batch['attention_mask'],
                                                        make_predictions=True)

                    # cls_softmax_predictions: (batch_size * max_length)
                    # cls_labels: (batch_size * max_length)
                    cls_softmax_predictions = \
                        (classifier_output.view(-1, classifier_output.shape[-1]).softmax(dim=-1)[:, -1:].squeeze() >
                         prediction_threshold).int()
                    cls_labels = data_batch['labels'].view(-1)
                    cls_labels_mask = (cls_labels != labels_ignore_idx).nonzero()
                    cls_y_true, cls_y_pred = cls_labels[cls_labels_mask].tolist(), \
                        cls_softmax_predictions[cls_labels_mask].tolist()
                    preds['cls']['y_true'] = preds['cls']['y_true'] + cls_y_true
                    preds['cls']['y_pred'] = preds['cls']['y_pred'] + cls_y_pred

        test_metrics = {'precision': precision_score(y_true=preds['cls']['y_true'], y_pred=preds['cls']['y_pred']),
                        'recall': recall_score(y_true=preds['cls']['y_true'], y_pred=preds['cls']['y_pred']),
                        'f1': f1_score(y_true=preds['cls']['y_true'], y_pred=preds['cls']['y_pred']),
                        'accuracy': accuracy_score(y_true=preds['cls']['y_true'], y_pred=preds['cls']['y_pred'])}
        if tracker is not None:
            tracker.log_metrics(epoch=0, split_name='test', metrics=test_metrics)

        return test_metrics
