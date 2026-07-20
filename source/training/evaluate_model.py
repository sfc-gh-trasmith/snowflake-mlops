"""Model evaluation utilities for fraud detection."""

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    """Comprehensive model evaluation with configurable threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_predictions": len(y_test),
        "fraud_rate": float(y_test.mean()),
    }
    return metrics


def check_quality_gate(metrics: dict, config: dict) -> tuple[bool, str]:
    """Check if model meets quality thresholds defined in pipeline config.

    Returns (passed: bool, reason: str)
    """
    min_auc = float(config.get("min_auc_roc", 0.85))
    min_precision = float(config.get("min_precision", 0.70))
    min_recall = float(config.get("min_recall", 0.60))

    failures = []
    if metrics["auc_roc"] < min_auc:
        failures.append(f"AUC-ROC {metrics['auc_roc']:.4f} < {min_auc}")
    if metrics["precision"] < min_precision:
        failures.append(f"Precision {metrics['precision']:.4f} < {min_precision}")
    if metrics["recall"] < min_recall:
        failures.append(f"Recall {metrics['recall']:.4f} < {min_recall}")

    if failures:
        return False, "; ".join(failures)
    return True, "All thresholds met"
