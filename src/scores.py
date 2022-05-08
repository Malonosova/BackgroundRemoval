import torch
import numpy as np
from src.u2net import normPRED


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, num=True, SMOOTH=1e-8):
    outputs = outputs.to(torch.int)
    labels = labels.to(torch.int)
    if num:
        outputs = outputs.squeeze(1).byte()
        labels = labels.squeeze(1).byte()
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou


def score_model(model, metric, data, tol_u2net=0.2, device=torch.device("cpu")):
    scores = []
    for X_batch, Y_label in data:
        Y_pred = normPRED(model(X_batch.to(device))[0].cpu())
        Y_pred = Y_pred >= tol_u2net
        scores.append(metric(Y_pred, Y_label).mean().item())
    return np.argsort(scores)


def score_all_metric(metric, mask_pred, mask):
    scores_fbeta = []
    scores_iou = []
    score_precision = []
    score_recall = []
    for n, y_pred in enumerate(mask_pred):
        y_label = mask[n].to(torch.int)
        scores_fbeta.append(metric[0](y_label.reshape(-1), y_pred.reshape(-1),  beta=0.3).mean().item())
        scores_iou.append(metric[1](y_pred.to(torch.int), y_label).mean().item())
        score_precision.append(metric[2](y_label.reshape(-1), y_pred.reshape(-1)))
        score_recall.append(metric[3](y_label.reshape(-1), y_pred.reshape(-1)))
    le = len(mask_pred)
    return sum(scores_fbeta)/le, sum(scores_iou)/le, sum(score_precision)/le, sum(score_recall)/le
