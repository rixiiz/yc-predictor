from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

@dataclass
class Metrics:
    acc: float
    f1: float
    auc: float
    cm: list[list[int]]

def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Metrics:
    y_pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, proba)) if len(set(y_true.tolist())) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return Metrics(acc=acc, f1=f1, auc=auc, cm=cm)

def metrics_to_dict(m: Metrics) -> Dict[str, Any]:
    return {"acc": m.acc, "f1": m.f1, "auc": m.auc, "confusion_matrix": m.cm}
