# QSAFE: Quantum Sensitivity-Aware Feature Encoding

**A hybrid quantum kernel ensemble framework for Drug-Induced Autoimmunity (DIA) prediction**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45-purple.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

QSAFE is a hybrid quantum-classical machine learning framework for predicting
Drug-Induced Autoimmunity (DIA) — a serious adverse drug reaction in which a
drug triggers an autoimmune response against the body's own tissues.

Classical ensemble methods trained on high-dimensional molecular descriptor
spaces systematically sacrifice sensitivity (the ability to detect DIA-positive
compounds) in favour of specificity. QSAFE addresses this limitation by:

1. Selecting a compact 6-feature subset per strategy (PCA / SHAP / MI / RFE)
2. Calibrating a scaling factor sf* via a novel two-stage kernel quality score J(sf) = -|μ - 0.5| + 0.5σ
3. Encoding features into a 6-qubit Parameterized Quantum Circuit (PQC)
4. Computing a fidelity quantum kernel K(xi, xj) = |⟨ψi|ψj⟩|²
5. Training a Quantum SVM (QSVM) per strategy
6. Combining the top-2 AUC-ranked strategies (SHAP + MI) via equal-weight soft voting — **this is QSAFE**

---

## Repository Structure

```
QSAFE/
│
├── notebooks/
│   ├── 01_QSAFE_pipeline.ipynb        # QSAFE quantum pipeline
│   │                                  # Feature selection, sf* calibration,
│   │                                  # QSVM training, ensemble voting
│   └── 02_classical_competitors.ipynb # Classical ML models on all 196 features
│                                      # LogReg, SVM-RBF, RF, XGBoost
│
├── src/
│   ├── quantum/
│   │   ├── feature_map.py             # 6-qubit PQC (H → RY → CNOT → RZZ)
│   │   ├── calibration.py            # Two-stage sf* calibration + J(sf) score
│   │   ├── kernel.py                  # Fidelity quantum kernel computation
│   │   └── qsvm.py                    # QSVM training with C grid search
│   │
│   ├── classical/
│   │   ├── feature_selection.py       # PCA, SHAP, MI, RFE strategies
│   │   └── models.py                  # Classical ML models
│   │
│   ├── ensemble/
│   │   └── voting.py                  # Hard, Soft, Weighted, Best-2, Best-3 voting
│   │
│   └── utils/
│       ├── metrics.py                 # ACC, SEN, SPE, F1, AUC, MCC
│       └── preprocessing.py          # SMOTE + StandardScaler pipeline
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## Dataset

The Guo et al. 2022 benchmark dataset is used:
- **Training set:** 477 compounds (118 DIA-positive, 359 DIA-negative)
- **Test set:** 120 compounds (30 DIA-positive, 90 DIA-negative)
- **Features:** 196 RDKit molecular descriptors

> The dataset is not included in this repository.
> Please refer to: Guo et al. (2022), *Frontiers in Immunology*,
> doi: 10.3389/fimmu.2022.1015409

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/NamitChawla/QSAFE.git
cd QSAFE
```

### 2. Create a virtual environment
```bash
python -m venv qsafe_env
source qsafe_env/bin/activate        # Linux/Mac
qsafe_env\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the QSAFE quantum pipeline
Open and run `notebooks/01_QSAFE_pipeline.ipynb` cell by cell.

The notebook covers:
- Data loading and preprocessing
- Feature selection (PCA, SHAP, MI, RFE) — 6 features per strategy
- Two-stage sf* calibration using J(sf) kernel quality score
- QSVM training with 5-fold stratified CV and C grid search
- Ensemble voting (Hard, Soft, Weighted, Best-2, Best-3)

### Run classical baseline models (196 features)
Open and run `notebooks/02_classical_competitors.ipynb`.

The notebook covers:
- LogReg, SVM-RBF, RF, XGBoost on all 196 RDKit features
- SMOTE class balancing inside each fold
- 5-fold stratified CV with identical preprocessing to QSAFE

---

## Quantum Circuit

The 6-qubit PQC consists of four layers:

```
Layer 1: H gate    — creates equal superposition across 64 basis states
Layer 2: RY(sf*·x) — encodes each molecular descriptor as a rotation angle
Layer 3: CNOT      — linear entanglement between adjacent qubits
Layer 4: RZZ(sf*·xi·xj) — encodes pairwise descriptor interactions
```

---

## Key Results

| Model | Features | ACC | AUC | SEN | SPE | MCC |
|---|---|---|---|---|---|---|
| **QSAFE** (QEnsemble-Best2: SHAP+MI) | 12 of 196 | 0.7950±0.019 | 0.8452±0.024 | **0.6467±0.044** | 0.8444±0.025 | 0.4754±0.047 |
| RF | 196 | 0.8183±0.022 | 0.8617±0.004 | 0.4000±0.067 | 0.9578±0.009 | 0.5222±0.072 |
| XGBoost | 196 | 0.8050±0.042 | 0.8438±0.034 | 0.4400±0.144 | 0.9266±0.013 | 0.5223±0.130 |
| SVM-RBF | 196 | 0.8150±0.020 | 0.7984±0.024 | 0.4133±0.069 | 0.9489±0.013 | 0.5258±0.063 |
| LogReg | 196 | 0.7333±0.047 | 0.7256±0.025 | 0.5400±0.055 | 0.7978±0.046 | 0.5052±0.069 |

**Key finding:** QSAFE achieves SEN = 0.6467 — a **61.7% relative improvement**
over Random Forest (SEN = 0.4000) — using 32× fewer features.

---

## Citation

If you use QSAFE in your research, please cite:

```bibtex
@article{qsafe2025,
  title   = {QSAFE: A Hybrid Quantum Kernel Ensemble Framework
             for Drug-Induced Autoimmunity Prediction},
  author  = {Namit et al.},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work was conducted at IIIT Una.
Quantum simulation performed using the Qiskit statevector simulator.
Dataset from Guo et al. (2022), Frontiers in Immunology.
