# API Overview

- `gnn_credit_fraud.data.simulate`: CLI to generate synthetic graphs.
- `gnn_credit_fraud.data.datasets.FinancialHeteroDataset`: Dataset API that produces PyG HeteroData.
- `gnn_credit_fraud.models.sage.SAGECreditRisk`: GraphSAGE for node classification.
- `gnn_credit_fraud.models.gat.GATFraudDetector`: GAT for edge classification.
- `gnn_credit_fraud.tasks.train_credit`: Train/eval credit risk model.
- `gnn_credit_fraud.tasks.train_fraud`: Train/eval fraud model.
