"""
Fidelity quantum kernel: K(xi, xj) = |<psi(xi)|psi(xj)>|^2
"""

import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from .feature_map import build_feature_map


def compute_kernel(X_train, X_test, n_qubits, sf_star):
    fm      = build_feature_map(n_qubits, sf_star)
    qk      = FidelityQuantumKernel(feature_map=fm)
    K_train = qk.evaluate(x_vec=X_train)
    K_test  = qk.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test
