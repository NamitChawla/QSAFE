"""
Parameterized Quantum Circuit (PQC) for QSAFE.

Four-layer circuit:
  Layer 1: Hadamard — superposition
  Layer 2: RY(sf * xi) — feature encoding
  Layer 3: CNOT (linear) — entanglement
  Layer 4: RZZ(sf * xi * xj) — pairwise interaction encoding
"""

from qiskit.circuit import QuantumCircuit, ParameterVector


def build_feature_map(n_qubits: int, scaling_factor: float) -> QuantumCircuit:
    params = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.ry(scaling_factor * params[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits - 1):
        qc.rzz(scaling_factor * params[i] * params[i + 1], i, i + 1)
    return qc
