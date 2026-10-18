"""Evaluation metrics per task type.

All metrics are computed from the raw output of the model's predict function
(``model.predict_function`` in config), so the same code scores a local model
and a registered champion called through ``ModelVersion.run()``.

  binary_classification  auc_roc, pr_auc, precision, recall, f1, log_loss
  multiclass             accuracy, f1_macro, log_loss (when probabilities)
  regression             rmse, mae, r2

The quality gate and champion comparison reference these names.
"""

import numpy as np

METRICS_BY_TASK = {
    "binary_classification": ("auc_roc", "pr_auc", "precision", "recall", "f1", "log_loss"),
    "multiclass": ("accuracy", "f1_macro", "log_loss"),
    "regression": ("rmse", "mae", "r2"),
}


def _as_2d(outputs) -> np.ndarray:
    arr = np.asarray(outputs, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def compute_metrics(task_type: str, y_true, outputs, threshold: float = 0.5) -> dict:
    """Compute metrics for a task type from predict-function outputs.

    Args:
        task_type: binary_classification | multiclass | regression
        y_true: ground-truth labels / targets
        outputs: predict_proba matrix (classification) or predictions
            (regression, or class labels when predict is used for classifiers)
        threshold: decision threshold for binary precision/recall/f1
    """
    from sklearn import metrics as m

    y_true = np.asarray(y_true)
    out = _as_2d(outputs)

    if task_type == "binary_classification":
        if out.shape[1] == 1 and not np.all(np.isin(out, (0, 1))):
            proba = out[:, 0]  # single positive-class score column
        elif out.shape[1] >= 2:
            proba = out[:, -1]  # predict_proba: last column is the positive class
        else:
            raise ValueError("binary_classification metrics need probability outputs (use predict_proba)")
        y_pred = (proba >= threshold).astype(int)
        return {
            "auc_roc": float(m.roc_auc_score(y_true, proba)),
            "pr_auc": float(m.average_precision_score(y_true, proba)),
            "precision": float(m.precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(m.recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(m.f1_score(y_true, y_pred, zero_division=0)),
            "log_loss": float(m.log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        }

    if task_type == "multiclass":
        if out.shape[1] > 1:
            y_pred = out.argmax(axis=1)
            return {
                "accuracy": float(m.accuracy_score(y_true, y_pred)),
                "f1_macro": float(m.f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "log_loss": float(m.log_loss(y_true, out, labels=list(range(out.shape[1])))),
            }
        y_pred = out[:, 0]
        return {
            "accuracy": float(m.accuracy_score(y_true, y_pred)),
            "f1_macro": float(m.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    if task_type == "regression":
        y_pred = out[:, 0]
        return {
            "rmse": float(np.sqrt(m.mean_squared_error(y_true, y_pred))),
            "mae": float(m.mean_absolute_error(y_true, y_pred)),
            "r2": float(m.r2_score(y_true, y_pred)),
        }

    raise ValueError(f"Unknown task_type: {task_type}")
