"""
Evaluation metrics for QSAFE.
"""

from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              confusion_matrix, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_prob, model_name=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1  = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    return {
        'Model': model_name, 'Accuracy': round(acc, 4),
        'Sensitivity': round(sen, 4), 'Specificity': round(spe, 4),
        'F1': round(f1, 4), 'AUC': round(auc, 4), 'MCC': round(mcc, 4),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
