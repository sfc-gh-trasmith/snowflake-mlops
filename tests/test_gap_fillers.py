"""Tests for model-agnostic metrics, output validation, data checks, serving
flags, gateway config handling, monitor SQL, and batch DAG wiring."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_engineering.data_checks import build_check_sql, evaluate_rules
from model_training.metrics import METRICS_BY_TASK, compute_metrics
from serving.signature import validate_outputs

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


# ─── Output validation ────────────────────────────────────────────────────────


def test_valid_probabilities_pass():
    assert validate_outputs("binary_classification", np.array([[0.3, 0.7], [0.6, 0.4]])) == []


def test_probabilities_not_summing_to_one_fail():
    problems = validate_outputs("binary_classification", np.array([[0.3, 0.3]]))
    assert any("sum to 1" in p for p in problems)


def test_null_predictions_fail():
    problems = validate_outputs("regression", np.array([1.0, np.nan]))
    assert any("null" in p for p in problems)


def test_regression_ignores_probability_checks():
    assert validate_outputs("regression", np.array([120.0, -5.0])) == []


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


# ─── Serving flags ────────────────────────────────────────────────────────────


def test_serving_flags_follow_config(monkeypatch):
    import config
    from utils import serving_flags

    monkeypatch.setitem(config.SERVING_CONFIG, "batch", {"enabled": True})
    monkeypatch.setitem(config.SERVING_CONFIG, "realtime", {"enabled": False})
    assert serving_flags.flags() == {"batch_enabled": "true", "realtime_enabled": "false", "monitor_enabled": "true"}


def test_serving_enabled_rejects_unknown_mode():
    from config import serving_enabled

    with pytest.raises(ValueError):
        serving_enabled("streaming")


def test_batch_input_defaults_to_feature_view():
    from config import FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION, batch_input_table

    assert batch_input_table("DB", "S") == f'DB.S."{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"'


# ─── Gateway config ───────────────────────────────────────────────────────────


def test_gateway_targets_parse_to_service_names():
    from apply_gateway_config import parse_targets_arg, service_name

    targets = parse_targets_arg("v6:80, V7:20")
    assert targets == [{"service": service_name("V6"), "weight": 80}, {"service": service_name("V7"), "weight": 20}]


@pytest.mark.parametrize(
    "targets, ok",
    [
        ([{"service": "A", "weight": 100}], True),
        ([{"service": "A", "weight": 80}, {"service": "B", "weight": 30}], False),
        ([{"service": "A", "weight": 50}, {"service": "A", "weight": 50}], False),
        ([], False),
    ],
)
def test_gateway_weight_validation(targets, ok):
    from apply_gateway_config import validate_weights

    assert (validate_weights(targets) == []) is ok


def test_gateway_config_roundtrip(tmp_path):
    from apply_gateway_config import load_config, update_config_file

    path = tmp_path / "gateway-config.yml"
    path.write_text("# header\n\ntargets:\n  - service: A\n    weight: 100\n")
    update_config_file([{"service": "A", "weight": 80}, {"service": "B", "weight": 20}], path)
    assert load_config(path)["targets"] == [{"service": "A", "weight": 80}, {"service": "B", "weight": 20}]
    assert path.read_text().startswith("# header")


def test_repo_gateway_config_is_valid():
    from apply_gateway_config import CONFIG_PATH, load_config, validate_weights

    assert CONFIG_PATH == ROOT / "gateway-config.yml"
    assert validate_weights(load_config()["targets"]) == []


def test_canary_split():
    from deploy_prod_service import canary_targets

    assert canary_targets("NEW", [], 20) == [{"service": "NEW", "weight": 100}]
    assert canary_targets("NEW", ["OLD"], 20) == [{"service": "OLD", "weight": 80}, {"service": "NEW", "weight": 20}]
    assert canary_targets("NEW", ["OLD"], 100) == [{"service": "NEW", "weight": 100}]


# ─── Monitor SQL ──────────────────────────────────────────────────────────────


def test_monitor_source_joins_actuals_and_filters_version():
    from monitoring.model_monitor import ACTUAL_COLUMN, labeled_select_sql

    sql = labeled_select_sql("DB.S.PRED", "V3")
    assert "LEFT JOIN" in sql and ACTUAL_COLUMN in sql
    assert "MODEL_VERSION = 'V3'" in sql


def test_monitor_ids_include_run_id():
    """Appended history repeats entity keys, so RUN_ID must be part of the ID."""
    from monitoring.model_monitor import monitor_columns_sql

    assert "'RUN_ID'" in monitor_columns_sql()


def test_alert_action_without_integration_only_logs(monkeypatch):
    import monitoring.model_monitor as mm

    monkeypatch.setitem(mm.MONITOR_CONFIG, "notification_integration", None)
    assert "SYSTEM$SEND_EMAIL" not in mm.alert_action_sql("DB", "S", "PSI", 0.25)
    monkeypatch.setitem(mm.MONITOR_CONFIG, "notification_integration", "EMAIL_INT")
    monkeypatch.setitem(mm.MONITOR_CONFIG, "notification_recipients", ["a@x.com"])
    assert "SYSTEM$SEND_EMAIL('EMAIL_INT'" in mm.alert_action_sql("DB", "S", "PSI", 0.25)


# ─── Batch DAG wiring ─────────────────────────────────────────────────────────


def test_batch_task_pins_env_and_imports_scorer():
    from config import resolve_env_config
    from pipeline.train_dag import runtime_env, sproc_definition

    cfg = resolve_env_config("prod")
    sql = sproc_definition("serving.batch_inference", runtime_env(cfg), "'@x/source.zip'", "'pyyaml'")
    assert cfg["database"] in sql and 'import_module("serving.batch_inference")' in sql


def test_serving_package_is_staged():
    from pipeline.train_dag import STAGED_PACKAGES

    assert "serving" in STAGED_PACKAGES  # training imports serving.signature


def test_batch_scorer_exposes_run():
    import inspect

    from serving import batch_inference

    assert list(inspect.signature(batch_inference.run).parameters) == ["session"]
