"""Tests for project configuration and quality gate logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import (
    COMPUTE_POOL,
    DATABASE,
    MIN_AUC_ROC,
    MIN_PRECISION,
    MIN_RECALL,
    ML_RUNTIME_DEPS,
    PIPELINE_CONFIG,
    SCHEMA,
    TRAINING_PARAMS,
    WAREHOUSE,
)
from quality_gate_and_register import check_quality_gate

# ─── Config Sanity Tests ──────────────────────────────────────────────────────


def test_config_values_are_strings():
    assert isinstance(DATABASE, str)
    assert isinstance(SCHEMA, str)
    assert isinstance(WAREHOUSE, str)
    assert isinstance(COMPUTE_POOL, str)


def test_pipeline_config_has_required_keys():
    required = ["model_name", "database", "schema", "warehouse", "compute_pool"]
    for key in required:
        assert key in PIPELINE_CONFIG, f"Missing required config key: {key}"


def test_training_params_are_numeric():
    for key in ["n_estimators", "learning_rate", "max_depth", "scale_pos_weight"]:
        assert key in TRAINING_PARAMS, f"Missing training param: {key}"
        assert isinstance(TRAINING_PARAMS[key], (int, float)), (
            f"Training param '{key}' should be numeric, got {type(TRAINING_PARAMS[key])}"
        )


def test_thresholds_are_reasonable():
    assert 0 < MIN_AUC_ROC <= 1.0
    assert 0 < MIN_PRECISION <= 1.0
    assert 0 < MIN_RECALL <= 1.0


def test_ml_runtime_deps_read_from_pyproject():
    """ML_RUNTIME_DEPS should match pyproject.toml [dependency-groups] ml-runtime."""
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        expected = tomllib.load(f)["dependency-groups"]["ml-runtime"]
    assert ML_RUNTIME_DEPS == expected
    assert len(ML_RUNTIME_DEPS) >= 2
    # Every entry should be pinned (contains ==)
    for dep in ML_RUNTIME_DEPS:
        assert "==" in dep, f"Dependency '{dep}' is not pinned"


# ─── Quality Gate Tests ───────────────────────────────────────────────────────


def test_quality_gate_all_pass():
    metrics = {"auc_roc": 0.95, "precision": 0.85, "recall": 0.80}
    passed, failures = check_quality_gate(metrics)
    assert passed is True
    assert failures == []


def test_quality_gate_auc_fails():
    metrics = {"auc_roc": 0.005, "precision": 0.85, "recall": 0.80}
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert len(failures) == 1
    assert "AUC-ROC" in failures[0]


def test_quality_gate_multiple_failures():
    metrics = {"auc_roc": 0.005, "precision": 0.005, "recall": 0.005}
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert len(failures) == 3


def test_quality_gate_missing_keys_default_to_zero():
    metrics = {}
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert len(failures) == 3


def test_quality_gate_boundary_values():
    metrics = {"auc_roc": MIN_AUC_ROC, "precision": MIN_PRECISION, "recall": MIN_RECALL}
    passed, failures = check_quality_gate(metrics)
    assert passed is True
    assert failures == []
