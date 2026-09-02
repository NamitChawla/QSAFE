"""
Quantum voting ensemble strategies for QSAFE.
"""

import numpy as np


def hard_voting(probs, preds):
    preds_stack = np.vstack(list(preds.values()))
    hard_pred   = (preds_stack.sum(axis=0) >= 3).astype(int)
    hard_prob   = np.mean(list(probs.values()), axis=0)
    return hard_pred, hard_prob


def soft_voting(probs):
    soft_prob = np.mean(list(probs.values()), axis=0)
    soft_pred = (soft_prob >= 0.5).astype(int)
    return soft_pred, soft_prob


def weighted_voting(probs, weights):
    strategies  = list(probs.keys())
    w_array     = np.array([weights[s] for s in strategies])
    w_norm      = w_array / w_array.sum()
    weight_prob = np.average([probs[s] for s in strategies], weights=w_norm, axis=0)
    weight_pred = (weight_prob >= 0.5).astype(int)
    return weight_pred, weight_prob


def best_k_voting(probs, auc_scores, k):
    top_k   = sorted(auc_scores, key=auc_scores.get, reverse=True)[:k]
    bk_prob = np.mean([probs[s] for s in top_k], axis=0)
    bk_pred = (bk_prob >= 0.5).astype(int)
    return bk_pred, bk_prob, top_k
