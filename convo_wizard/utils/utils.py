import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, accuracy_score


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Setting `torch.backends.cudnn.benchmark = False` slows down training.
    # Reference: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.backends.cudnn.benchmark = True


def device_mapper(input_tensor, device):
    return input_tensor.to(device) if not input_tensor.device == device else input_tensor


def find_best_threshold_using_prc(y_true, y_pred_proba, criterion='acc'):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred_proba)
    criterion_scores = []
    if criterion == 'acc':
        for threshold in thresholds:
            acc_score = accuracy_score(y_true=y_true, y_pred=[pred_proba >= threshold for pred_proba in y_pred_proba])
            criterion_scores.append(acc_score)
    else:
        criterion = 'f1'  # use F1-scores to choose the best threshold
        criterion_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_threshold_idx = np.nanargmax(criterion_scores)
    logging.info(
        f'best threshold: {thresholds[best_threshold_idx]}, {criterion}: {criterion_scores[best_threshold_idx]}, '
        f'precision: {precisions[best_threshold_idx]}, recall: {recalls[best_threshold_idx]}')
    return thresholds[best_threshold_idx]
