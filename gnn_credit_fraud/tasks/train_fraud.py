import pytorch_lightning as pl
import torch, argparse
from gnn_credit_fraud.data.datasets import FinancialHeteroDataset
from gnn_credit_fraud.models.gat import GATFraudDetector
from gnn_credit_fraud.utils.seed import seed_everything
from gnn_credit_fraud.utils.metrics import binary_classification_metrics

class FraudModule(pl.LightningModule):
    def __init__(self, data_root: str, hidden_dim: int = 64, lr: float = 1e-3, max_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        ds = FinancialHeteroDataset(data_root)
        data = ds.hetero_graph()
        self.x_b = data["borrower"].x
        self.y_edge = data[("borrower","purchased_at","merchant")].edge_label.float()
        self.x_m = data["merchant"].x
        self.edge_index = data[("borrower","purchased_at","merchant")].edge_index
        self.model = GATFraudDetector(self.x_b.size(1), self.x_m.size(1), hidden_dim)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        logits = self.model(self.x_b, self.x_m, self.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, self.y_edge)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            prob = torch.sigmoid(self.model(self.x_b, self.x_m, self.edge_index)).cpu().numpy()
        y = self.y_edge.cpu().numpy()
        m = binary_classification_metrics(y, prob)
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
    model = FraudModule(args.data_root, args.hidden_dim, args.lr, args.max_epochs)
    trainer.fit(model)
