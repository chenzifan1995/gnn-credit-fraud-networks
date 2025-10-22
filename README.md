# Graph Neural Networks for Credit Risk Modeling and Fraud Detection in Financial Networks

A research-grade, production-friendly repository that demonstrates **GNN pipelines** for:
- **Credit risk modeling** (node classification: predict borrower default).
- **Fraud detection** (edge classification: detect suspicious transactions).

The project builds **heterogeneous financial graphs** (borrowers, loans, merchants, transactions)
and trains **GraphSAGE/GAT** models with configurable experiments via **Hydra**.

## Highlights
- Synthetic, **privacy-safe** graph generator that mimics real-world credit + transaction networks.
- Unified training loop using **PyTorch Lightning** and **PyTorch Geometric**.
- Separate tasks: `credit_risk` (node), `fraud_detection` (edge).
- Reproducible experiments via `configs/` and `Makefile` targets.
- CI-tested unit tests and style checks.
- Dockerfile for containerized runs.

## Quickstart
```bash
# 1) (Optional) Use a fresh Python env (3.10+ recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies (PyTorch & torch-geometric wheels may require platform-specific URLs)
pip install -r requirements.txt

# 3) Generate synthetic data
python -m gnn_credit_fraud.data.simulate --out data/simulated --seed 42 --n_borrowers 5000

# 4) Train credit-risk node classifier
python -m gnn_credit_fraud.tasks.train_credit task.credit_risk.max_epochs=3

# 5) Train fraud-detection edge classifier
python -m gnn_credit_fraud.tasks.train_fraud task.fraud_detection.max_epochs=3
```

> **Note on PyTorch Geometric**: visit https://pytorch-geometric.readthedocs.io for platform-specific install commands.

## Repository Layout
```
gnn-credit-fraud-networks/
├── gnn_credit_fraud/         # Python package (source)
│   ├── data/                 # Simulation & dataset loaders
│   ├── models/               # GNN architectures
│   ├── tasks/                # Training/eval scripts for tasks
│   ├── utils/                # Helpers (metrics, logging, configs)
│   └── __init__.py
├── configs/                  # Hydra configs for experiments
├── docs/                     # Design notes and API docs
├── examples/                 # Schemas and sample queries
├── notebooks/                # EDA & ablation study notebooks
├── tests/                    # Unit tests
├── .github/workflows/ci.yml  # CI pipeline (pytest + lint)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── Makefile
├── LICENSE
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
└── README.md
```

## Datasets (Synthetic)
This repo uses generated data to avoid PII and sharing restrictions:
- **Borrowers** with demographics + credit attributes
- **Merchants** with category codes
- **Loans** linking borrowers
- **Transactions** edges linking borrowers ↔ merchants with labels for fraud

You can replace the generator with your own loaders by implementing the interfaces in `gnn_credit_fraud/data/datasets.py`.

## Experiments
Hydra config examples live in `configs/`. To override any parameter, pass it on the CLI:
```bash
python -m gnn_credit_fraud.tasks.train_credit optimizer.lr=2e-4 model.hidden_dim=128
```

## Results & Reproducibility
- Deterministic seeds via `seed_everything`.
- Train/val/test splits with fixed seeds.
- Metrics: AUROC/AP/F1 for classification; AUC-PR emphasized for imbalanced fraud.

## Citation
If you use this repository, please cite it:
```bibtex
@software{gnn_credit_fraud_networks,
  title={Graph Neural Networks for Credit Risk Modeling and Fraud Detection in Financial Networks},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/gnn-credit-fraud-networks}
}
```
