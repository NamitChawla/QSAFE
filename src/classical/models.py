"""
Classical ML models matching QSAFE pipeline hyperparameters.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42


def get_classical_models():
    return {
        'LogReg':  LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'SVM-RBF': SVC(kernel='rbf', probability=True, C=10, random_state=RANDOM_SEED),
        'RF':      RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                  use_label_encoder=False, eval_metric='logloss', verbosity=0)
    }
