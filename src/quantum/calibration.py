"""
Two-stage scaling factor (sf*) calibration for QSAFE.

J(sf) = -|mu - 0.5| + 0.5 * sigma
"""

import numpy as np
import pandas as pd
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from .feature_map import build_feature_map

RANDOM_SEED = 42


def kernel_quality_score(K: np.ndarray) -> tuple:
    off_diag = K[~np.eye(len(K), dtype=bool)]
    mu  = off_diag.mean()
    sig = off_diag.std()
    J   = -abs(mu - 0.5) + 0.5 * sig
    return mu, sig, J


def calibrate_scaling_factor(
    X_sample: np.ndarray,
    n_qubits: int,
    verbose: bool = False
) -> tuple:
    X_cal = X_sample[:50] if len(X_sample) >= 50 else X_sample
    coarse_candidates = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    coarse_results = []

    for sf in coarse_candidates:
        fm  = build_feature_map(n_qubits, sf)
        qk  = FidelityQuantumKernel(feature_map=fm)
        K   = qk.evaluate(x_vec=X_cal)
        mu, sig, J = kernel_quality_score(K)
        coarse_results.append({'sf': sf, 'mean': mu, 'std': sig, 'score': J})
        if verbose:
            print(f"  [coarse] sf={sf:.2f}  mean={mu:.4f}  std={sig:.4f}  score={J:.4f}")

    df_coarse      = pd.DataFrame(coarse_results)
    best_coarse_sf = df_coarse.loc[df_coarse['score'].idxmax(), 'sf']
    idx   = coarse_candidates.index(best_coarse_sf)
    lower = coarse_candidates[max(0, idx - 1)]
    upper = coarse_candidates[min(len(coarse_candidates) - 1, idx + 1)]
    fine_candidates = sorted(set(np.round(np.linspace(lower, upper, 7), 2).tolist()))
    fine_results = []

    for sf in fine_candidates:
        fm  = build_feature_map(n_qubits, sf)
        qk  = FidelityQuantumKernel(feature_map=fm)
        K   = qk.evaluate(x_vec=X_cal)
        mu, sig, J = kernel_quality_score(K)
        fine_results.append({'sf': sf, 'mean': mu, 'std': sig, 'score': J})
        if verbose:
            print(f"  [fine]   sf={sf:.2f}  mean={mu:.4f}  std={sig:.4f}  score={J:.4f}")

    df_fine = pd.DataFrame(fine_results)
    sf_star = df_fine.loc[df_fine['score'].idxmax(), 'sf']
    return sf_star, df_coarse, df_fine
