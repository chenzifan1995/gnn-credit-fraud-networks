from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
    }
