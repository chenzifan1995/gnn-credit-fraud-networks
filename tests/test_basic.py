import os, json, numpy as np
from gnn_credit_fraud.data.simulate import generate_synthetic
from gnn_credit_fraud.data.datasets import FinancialHeteroDataset

def test_simulation_and_dataset(tmp_path):
    out = tmp_path / "sim"
    generate_synthetic(str(out), seed=1, n_borrowers=200, n_merchants=50)
    ds = FinancialHeteroDataset(str(out))
    data = ds.hetero_graph()
    assert data["borrower"].x.shape[0] == 200
    assert data["merchant"].x.shape[0] == 50
    ei = data[("borrower","purchased_at","merchant")].edge_index
    assert ei.shape[0] == 2
