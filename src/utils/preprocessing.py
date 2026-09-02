"""
Preprocessing utilities for QSAFE.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42


def smote_balance(X, y, fold=0):
    smote = SMOTE(random_state=RANDOM_SEED + fold)
    return smote.fit_resample(X, y)


def scale_features(X_train, X_test):
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler
