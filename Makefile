.PHONY: setup data credit fraud test lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python -m gnn_credit_fraud.data.simulate --out data/simulated --seed 42 --n_borrowers 2000

credit:
	python -m gnn_credit_fraud.tasks.train_credit

fraud:
	python -m gnn_credit_fraud.tasks.train_fraud

test:
	pytest -q

lint:
	python -m pip install flake8 && flake8 gnn_credit_fraud
