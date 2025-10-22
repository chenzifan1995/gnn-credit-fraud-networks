import json, os, numpy as np, torch
from torch.utils.data import Dataset
try:
    from torch_geometric.data import HeteroData
except Exception:
    HeteroData = object

class FinancialHeteroDataset(Dataset):
    """Loads synthetic JSONL into a simplistic hetero-graph (PyG HeteroData).
    If torch-geometric is unavailable, raises a runtime error at __getitem__.
    """
    def __init__(self, root: str, split: str = "train", seed: int = 42):
        self.root = root
        self.split = split
        self._load()

    def _load(self):
        self.borrowers = [json.loads(l) for l in open(os.path.join(self.root, "borrowers.jsonl"))]
        self.merchants = [json.loads(l) for l in open(os.path.join(self.root, "merchants.jsonl"))]
        self.transactions = [json.loads(l) for l in open(os.path.join(self.root, "transactions.jsonl"))]

        # deterministic split on borrower_id
        ids = np.array([b["borrower_id"] for b in self.borrowers])
        rng = np.random.default_rng(42)
        rng.shuffle(ids)
        n = len(ids)
        tr, va = int(0.7*n), int(0.85*n)
        self.train_ids = set(ids[:tr])
        self.val_ids = set(ids[tr:va])
        self.test_ids = set(ids[va:])

    def _node_features_borrower(self):
        import numpy as np
        X = []
        y = []
        for b in self.borrowers:
            X.append([b["age"], b["income"], b["region"], b["credit_score"], b["delinquency_count"]])
            y.append(b["label_default"])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

    def hetero_graph(self):
        try:
            from torch_geometric.data import HeteroData
        except Exception:
            raise RuntimeError("torch-geometric not installed; cannot build HeteroData.")
        import torch

        Xb, yb = self._node_features_borrower()
        data = HeteroData()
        data["borrower"].x = torch.tensor(Xb)
        data["borrower"].y = torch.tensor(yb)

        Xm = []
        for m in self.merchants:
            Xm.append([m["mcc"], m["risk_score"]])
        data["merchant"].x = torch.tensor(np.asarray(Xm, dtype=np.float32))

        # Edges borrower<->merchant for transactions
        src, dst, fraud = [], [], []
        for t in self.transactions:
            src.append(t["borrower_id"]); dst.append(t["merchant_id"]); fraud.append(t["is_fraud"])
        data["borrower", "purchased_at", "merchant"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["borrower", "purchased_at", "merchant"].edge_label = torch.tensor(fraud, dtype=torch.long)

        return data
