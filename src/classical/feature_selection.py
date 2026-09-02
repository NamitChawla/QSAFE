"""
Feature selection strategies: PCA, SHAP, MI, RFE
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

RANDOM_SEED = 42


def select_features_once(strategy, X_train_df, y_train, n_features):
    smote = SMOTE(random_state=RANDOM_SEED)
    X_sm, y_sm = smote.fit_resample(X_train_df, y_train)
    X_sm_df    = pd.DataFrame(X_sm, columns=X_train_df.columns)
    scaler     = StandardScaler()
    X_sm_sc    = pd.DataFrame(scaler.fit_transform(X_sm_df), columns=X_sm_df.columns)

    if strategy == 'PCA':
        pca   = PCA(n_components=n_features, random_state=RANDOM_SEED)
        pca.fit(X_sm_sc)
        var   = pca.explained_variance_ratio_
        names = [f"PC{i+1}({var[i]*100:.1f}%)" for i in range(n_features)]
        return {'strategy': 'PCA', 'feat_names': names, 'scaler': scaler, 'pca': pca}
    elif strategy == 'SHAP':
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_sm_sc, y_sm)
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_sm_sc)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        elif shap_vals.ndim == 3: shap_vals = shap_vals[:, :, 1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:n_features]
        names    = [X_sm_sc.columns[i] for i in top_idx]
    elif strategy == 'MI':
        mi_scores = mutual_info_classif(X_sm_sc, y_sm, random_state=RANDOM_SEED)
        top_idx   = np.argsort(mi_scores)[::-1][:n_features]
        names     = [X_sm_sc.columns[i] for i in top_idx]
    elif strategy == 'RFE':
        estimator = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=5)
        rfe.fit(X_sm_sc, y_sm)
        names = X_sm_sc.columns[rfe.support_].tolist()
    return {'strategy': strategy, 'feat_names': names, 'scaler': scaler, 'pca': None}


def apply_fixed_transform(fs_artifact, X_df):
    X_sc = pd.DataFrame(fs_artifact['scaler'].transform(X_df), columns=X_df.columns)
    if fs_artifact['strategy'] == 'PCA':
        return fs_artifact['pca'].transform(X_sc)
    return X_sc[fs_artifact['feat_names']].values
