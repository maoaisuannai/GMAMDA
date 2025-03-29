import numpy as np
import torch
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve, roc_auc_score)


def specificity_score(y_true, y_pred):
    """计算特异性（真阴性率）"""
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def get_metric(y_true, y_pred, y_prob):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    Auc = auc(fpr, tpr)

    precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
    Aupr = auc(recall1, precision1)
    spe = specificity_score(y_true, y_pred)

    return Auc, Aupr, accuracy, precision, recall, f1, mcc, spe