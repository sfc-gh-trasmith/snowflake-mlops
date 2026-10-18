"""Tests for project configuration and quality gate logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import (
    CHAMPION_CONFIG,
    COMPUTE_POOL,
    DATABASE,
    GATE_RULES,
    ML_RUNTIME_DEPS,
    PIPELINE_CONFIG,
    SCHEMA,
    TASK_TYPE,
    TRAINING_PARAMS,
    WAREHOUSE,
)
from evaluate.gate import check_rules, compare_champion
from model_training.metrics import METRICS_BY_TASK

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


def test_gate_rules_reference_metrics_of_the_task_type():
    """A typo in a gate metric name would make the gate fail every run."""
    for name, rule in GATE_RULES.items():
        assert name in METRICS_BY_TASK[TASK_TYPE], f"Gate metric '{name}' is not produced for {TASK_TYPE}"
        assert set(rule) <= {"min", "max"}, f"Gate rule for '{name}' must use min/max: {rule}"


def test_champion_metric_is_produced():
    if CHAMPION_CONFIG.get("enabled"):
        assert CHAMPION_CONFIG["metric"] in METRICS_BY_TASK[TASK_TYPE]


def test_ml_runtime_deps_read_from_pyproject():
    """ML_RUNTIME_DEPS should match pyproject.toml [dependency-groups] ml-runtime."""
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        expected = tomllib.load(f)["dependency-groups"]["ml-runtime"]
    assert ML_RUNTIME_DEPS == expected
    assert len(ML_RUNTIME_DEPS) >= 2
    for dep in ML_RUNTIME_DEPS:
        assert "==" in dep, f"Dependency '{dep}' is not pinned"


# ─── Quality gate rules ───────────────────────────────────────────────────────


def test_rules_all_pass():
    assert check_rules({"auc_roc": 0.95, "recall": 0.8}, {"auc_roc": {"min": 0.9}, "recall": {"min": 0.5}}) == []


def test_rules_min_and_max():
    rules = {"auc_roc": {"min": 0.9}, "rmse": {"max": 10}}
    failures = check_rules({"auc_roc": 0.5, "rmse": 12.0}, rules)
    assert len(failures) == 2
    assert "auc_roc" in failures[0] and "rmse" in failures[1]


def test_rules_boundary_values_pass():
    assert check_rules({"auc_roc": 0.9, "rmse": 10.0}, {"auc_roc": {"min": 0.9}, "rmse": {"max": 10}}) == []


def test_rules_missing_metric_fails():
    failures = check_rules({}, {"auc_roc": {"min": 0.5}})
    assert failures and "not produced" in failures[0]


# ─── Champion / challenger ────────────────────────────────────────────────────

CHAMP = {"available": True, "version": "V9", "metrics": {"auc_roc": 0.80, "rmse": 5.0}}


def test_champion_disabled_passes():
    passed, _ = compare_champion({"auc_roc": 0.1}, CHAMP, {"enabled": False})
    assert passed


def test_no_champion_passes():
    passed, note = compare_champion({"auc_roc": 0.1}, {"available": False, "reason": "first deploy"}, {"enabled": True})
    assert passed and "first deploy" in note


@pytest.mark.parametrize(
    "candidate, margin, expected",
    [(0.85, 0.0, True), (0.79, 0.0, False), (0.795, -0.01, True), (0.81, 0.02, False)],
)
def test_champion_higher_is_better(candidate, margin, expected):
    cfg = {"enabled": True, "metric": "auc_roc", "higher_is_better": True, "min_improvement": margin}
    passed, _ = compare_champion({"auc_roc": candidate}, CHAMP, cfg)
    assert passed is expected


def test_champion_lower_is_better():
    cfg = {"enabled": True, "metric": "rmse", "higher_is_better": False, "min_improvement": 0.0}
    assert compare_champion({"rmse": 4.0}, CHAMP, cfg)[0] is True
    assert compare_champion({"rmse": 6.0}, CHAMP, cfg)[0] is False


def test_champion_missing_metric_fails():
    cfg = {"enabled": True, "metric": "f1"}
    assert compare_champion({"auc_roc": 0.9}, CHAMP, cfg)[0] is False
