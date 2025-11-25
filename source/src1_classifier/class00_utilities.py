import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_confusion(y_true, y_pred_logits, threshold=0.5):
    """y_true: tensor or array of 0/1, y_pred_logits: logits -> apply sigmoid"""
    probs = torch.sigmoid(torch.tensor(y_pred_logits))
    preds = (probs >= threshold).int()
    y_true = torch.tensor(y_true).int()
    TP = int(((preds == 1) & (y_true == 1)).sum())
    TN = int(((preds == 0) & (y_true == 0)).sum())
    FP = int(((preds == 1) & (y_true == 0)).sum())
    FN = int(((preds == 0) & (y_true == 1)).sum())
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
