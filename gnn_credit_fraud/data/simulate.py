import os, argparse, json, random, math
import numpy as np

def generate_synthetic(out_dir: str, seed: int = 42, n_borrowers: int = 5000, n_merchants: int = 1200):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    borrowers = []
    for i in range(n_borrowers):
        income = float(rng.normal(60000, 20000))
        credit_score = float(np.clip(rng.normal(680, 70), 300, 850))
        delinquency = int(max(0, rng.poisson(0.5)))
        default_prob = max(0.01, min(0.7, 0.25 * (700-credit_score)/400 + 0.15 * (delinquency/5)))
        borrowers.append({
            "borrower_id": i,
            "age": int(np.clip(rng.normal(38, 10), 18, 80)),
            "income": income,
            "region": int(rng.integers(0, 20)),
            "credit_score": credit_score,
            "delinquency_count": delinquency,
            "label_default": int(rng.random() < default_prob)
        })

    merchants = []
    for j in range(n_merchants):
        mcc = int(rng.integers(1000, 6000))
        risk = float(np.clip(rng.beta(2, 8), 0, 1))
        merchants.append({"merchant_id": j, "mcc": mcc, "risk_score": risk})

    # Transactions ~ borrowers x poisson merchants
    transactions = []
    tx_id = 0
    for b in borrowers:
        k = int(np.clip(np.random.poisson(30), 5, 120))
        for _ in range(k):
            m = int(rng.integers(0, n_merchants))
            amount = float(max(1, rng.lognormal(3, 0.8)))
            ts = int(rng.integers(1_600_000_000, 1_700_000_000))
            # Fraud depends on merchant risk & borrower delinquency
            p_fraud = 0.02 + 0.4*merchants[m]["risk_score"] + 0.02*b["delinquency_count"]
            is_fraud = int(rng.random() < min(0.95, p_fraud))
            transactions.append({
                "tx_id": tx_id, "borrower_id": b["borrower_id"], "merchant_id": m,
                "amount": amount, "timestamp": ts, "is_fraud": is_fraud
            })
            tx_id += 1

    # Save
    with open(os.path.join(out_dir, "borrowers.jsonl"), "w") as f:
        for row in borrowers: f.write(json.dumps(row)+"\n")
    with open(os.path.join(out_dir, "merchants.jsonl"), "w") as f:
        for row in merchants: f.write(json.dumps(row)+"\n")
    with open(os.path.join(out_dir, "transactions.jsonl"), "w") as f:
        for row in transactions: f.write(json.dumps(row)+"\n")

    meta = {"n_borrowers": n_borrowers, "n_merchants": n_merchants, "n_transactions": len(transactions), "seed": seed}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote synthetic dataset to {out_dir} with {len(transactions)} transactions.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_borrowers", type=int, default=5000)
    p.add_argument("--n_merchants", type=int, default=1200)
    args = p.parse_args()
    generate_synthetic(args.out, args.seed, args.n_borrowers, args.n_merchants)
