# System Design

We model a **heterogeneous financial network**:
- Nodes: `Borrower`, `Merchant`, `Loan`
- Edges: `BORROWED` (Borrower→Loan), `PURCHASED_AT` (Borrower↔Merchant), `ISSUED_BY` (Loan→Borrower)
- Edge labels for **fraud** on `PURCHASED_AT`
- Node labels for **credit default** on `Borrower`

Two tasks:
1. **Credit Risk** (node classification): predict whether a borrower will default.
2. **Fraud Detection** (edge classification): predict whether a transaction is fraudulent.
