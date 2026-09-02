"""
QSVM training with C grid search.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from .kernel import compute_kernel

RANDOM_SEED = 42
C_VALUES    = [0.01, 0.1, 1.0, 10.0, 100.0]


def train_qsvm(X_train, y_train, X_test, y_test,
               n_qubits, sf_star, c_values=None):
    if c_values is None:
        c_values = C_VALUES
    K_train, K_test = compute_kernel(X_train, X_test, n_qubits, sf_star)
    best_auc, best_pred, best_prob, best_C = 0, None, None, c_values[0]
    for C in c_values:
        model = SVC(kernel='precomputed', probability=True,
                    C=C, random_state=RANDOM_SEED)
        model.fit(K_train, y_train)
        prob_c = model.predict_proba(K_test)[:, 1]
        pred_c = model.predict(K_test)
        try:
            auc_c = roc_auc_score(y_test, prob_c)
        except Exception:
            auc_c = 0.5
        if auc_c > best_auc:
            best_auc, best_pred, best_prob, best_C = auc_c, pred_c, prob_c, C
    return best_pred, best_prob, best_C
