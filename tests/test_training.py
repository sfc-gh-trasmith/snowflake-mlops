"""Tests for task-type metrics and upstream data checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "deploy"))

from feature_engineering.data_checks import build_check_sql, evaluate_rules
from model_training.metrics import METRICS_BY_TASK, compute_metrics

# ─── Metrics per task type ────────────────────────────────────────────────────


def test_binary_metrics_from_predict_proba():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    m = compute_metrics("binary_classification", y, proba)
    assert set(m) == set(METRICS_BY_TASK["binary_classification"])
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(1.0)


def test_binary_metrics_reject_hard_labels():
    with pytest.raises(ValueError, match="predict_proba"):
        compute_metrics("binary_classification", [0, 1], np.array([0, 1]))


def test_multiclass_metrics():
    y = np.array([0, 1, 2])
    proba = np.eye(3) * 0.8 + 0.2 / 3
    m = compute_metrics("multiclass", y, proba)
    assert m["accuracy"] == pytest.approx(1.0)
    assert "log_loss" in m


def test_regression_metrics():
    m = compute_metrics("regression", [1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))


def test_unknown_task_type():
    with pytest.raises(ValueError):
        compute_metrics("ranking", [1], [1])


# ─── Data checks ──────────────────────────────────────────────────────────────

RULES = {"min_rows": 100, "max_null_pct": {"ID": 0, "LABEL": 5}, "freshness_column": "TS", "max_age_hours": 24}


def test_check_sql_includes_each_rule():
    sql = build_check_sql("DB.S.T", RULES)
    assert "COUNT(*) AS ROW_COUNT" in sql
    assert "NULL_PCT_0" in sql and "NULL_PCT_1" in sql
    assert "AGE_HOURS" in sql and "FROM DB.S.T" in sql


def test_check_sql_skips_freshness_without_max_age():
    assert "AGE_HOURS" not in build_check_sql("T", {"freshness_column": "TS", "max_age_hours": None})


def test_rules_pass():
    stats = {"ROW_COUNT": 500, "NULL_PCT_0": 0, "NULL_PCT_1": 2.0, "AGE_HOURS": 3}
    assert evaluate_rules("T", RULES, stats) == []


def test_rules_report_every_failure():
    stats = {"ROW_COUNT": 10, "NULL_PCT_0": 1.0, "NULL_PCT_1": 9.0, "AGE_HOURS": 48}
    failures = evaluate_rules("T", RULES, stats)
    assert len(failures) == 4
