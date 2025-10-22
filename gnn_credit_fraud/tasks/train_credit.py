import pytorch_lightning as pl
import torch, argparse
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

from gnn_credit_fraud.data.datasets import FinancialHeteroDataset
from gnn_credit_fraud.models.sage import SAGECreditRisk
from gnn_credit_fraud.utils.seed import seed_everything
from gnn_credit_fraud.utils.metrics import binary_classification_metrics

class CreditModule(pl.LightningModule):
    def __init__(self, data_root: str, hidden_dim: int = 64, lr: float = 1e-3, max_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = FinancialHeteroDataset(data_root)
        data = self.dataset.hetero_graph()
        self.data = data
        in_dim = data["borrower"].x.size(1)
        # create simple homogeneous graph on borrower nodes via k-NN (approx: chain)
        n = data["borrower"].x.size(0)
        edge_index = torch.stack([torch.arange(0, n-1), torch.arange(1, n)], dim=0)
        self.edge_index = to_undirected(edge_index)
        self.model = SAGECreditRisk(in_dim, hidden_dim)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x = self.data["borrower"].x
        y = self.data["borrower"].y.float()
        logits = self.model(x, self.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.data["borrower"].x
        y = self.data["borrower"].y.cpu().numpy()
        with torch.no_grad():
            proba = torch.sigmoid(self.model(x, self.edge_index)).cpu().numpy()
        m = binary_classification_metrics(y, proba)
        for k,v in m.items():
            self.log(f"val_{k}", v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/simulated")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_epochs", type=int, default=5)
    args = ap.parse_args()
    seed_everything(42)
    trainer = pl.Trainer(max_epochs=args.max_epochs, enable_checkpointing=False, logger=False)
    model = CreditModule(args.data_root, args.hidden_dim, args.lr, args.max_epochs)
    trainer.fit(model)
