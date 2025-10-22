-- Example minimal schema for a financial network
CREATE TABLE borrowers (
  borrower_id BIGINT PRIMARY KEY,
  age INT, income FLOAT, region TEXT, credit_score FLOAT, delinquency_count INT
);

CREATE TABLE merchants (
  merchant_id BIGINT PRIMARY KEY,
  mcc INT, risk_score FLOAT
);

CREATE TABLE loans (
  loan_id BIGINT PRIMARY KEY,
  borrower_id BIGINT REFERENCES borrowers(borrower_id),
  amount FLOAT, interest_rate FLOAT, term_months INT, status TEXT
);

CREATE TABLE transactions (
  tx_id BIGINT PRIMARY KEY,
  borrower_id BIGINT REFERENCES borrowers(borrower_id),
  merchant_id BIGINT REFERENCES merchants(merchant_id),
  amount FLOAT, timestamp BIGINT, is_fraud INT
);
