"""Tests for output validation, serving flags, the gateway split, monitor SQL,
batch DAG wiring, service cleanup, and the REST smoke test helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "deploy"))

from serving.signature import validate_outputs

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
        ([{"service": "A", "weight": 100}, {"service": "B", "weight": 0}], False),
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
    """Shipped config is either empty (fresh template) or a valid split."""
    from apply_gateway_config import CONFIG_PATH, load_config, validate_weights

    assert CONFIG_PATH == ROOT / "deploy" / "gateway-config.yml"
    targets = load_config()["targets"]
    assert targets == [] or validate_weights(targets) == []


@pytest.mark.parametrize("body", ["targets: []\n", "targets:\n", "# only a comment\n"])
def test_empty_gateway_config_is_a_noop(tmp_path, monkeypatch, body):
    """Before the first PROD deploy, a push touching the file must not fail."""
    import apply_gateway_config as agc

    path = tmp_path / "gateway-config.yml"
    path.write_text(body)
    monkeypatch.setattr(agc, "CONFIG_PATH", path)
    monkeypatch.setattr(sys, "argv", ["apply_gateway_config.py"])
    assert agc.load_config()["targets"] == []
    agc.main()  # returns before opening a Snowflake session


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


# ─── Service cleanup ──────────────────────────────────────────────────────────


class _FakeSession:
    """Returns canned rows per SQL prefix and records every statement."""

    def __init__(self, responses):
        self.responses, self.statements = responses, []

    def sql(self, query):
        self.statements.append(" ".join(query.split()))
        rows = next((r for prefix, r in self.responses.items() if query.strip().startswith(prefix)), [])
        return type("R", (), {"collect": lambda _self: rows})()


GATEWAY_SPEC = """spec:
  type: traffic_split
  split_type: custom
  targets:
    - type: endpoint
      value: DB.S.M_SERVICE_V4!inference
      weight: 80
    - type: endpoint
      value: DB.S.M_SERVICE_V5!inference
      weight: 20
"""


def test_gateway_targets_parse_from_spec():
    from apply_gateway_config import get_current_gateway_services, get_current_gateway_targets

    session = _FakeSession({"DESC GATEWAY": [{"spec": GATEWAY_SPEC}]})
    assert get_current_gateway_targets(session, "DB", "S", "GW") == [
        {"service": "M_SERVICE_V4", "weight": 80},
        {"service": "M_SERVICE_V5", "weight": 20},
    ]
    assert get_current_gateway_services(session, "DB", "S", "GW") == ["M_SERVICE_V4", "M_SERVICE_V5"]


def test_services_without_traffic_are_unused():
    from apply_gateway_config import unused_services

    existing = ["M_SERVICE_V3", "M_SERVICE_V4", "M_SERVICE_V5"]
    assert unused_services(existing, [{"service": "M_SERVICE_V5", "weight": 100}]) == ["M_SERVICE_V3", "M_SERVICE_V4"]
    assert unused_services(existing, [{"service": s, "weight": 1} for s in existing]) == []


def test_cleanup_drops_every_service_outside_the_split():
    from apply_gateway_config import MODEL_NAME, cleanup_unused_services

    old, live = f"{MODEL_NAME}_SERVICE_V3", f"{MODEL_NAME}_SERVICE_V4"
    session = _FakeSession({"SHOW SERVICES": [{"name": old}, {"name": live}]})
    assert cleanup_unused_services(session, "DB", "S", [{"service": live, "weight": 100}]) == [old]
    assert session.statements[-1] == f"DROP SERVICE IF EXISTS DB.S.{old}"


# ─── REST smoke test ──────────────────────────────────────────────────────────


def test_rest_path_and_body():
    import pandas as pd
    from smoke_test_endpoint import method_path, rest_body

    assert method_path("predict_proba") == "/predict-proba"
    body = rest_body(pd.DataFrame({"A": [1, 2], "B": [0.5, 1.5]}))
    assert body == {"dataframe_split": {"columns": ["A", "B"], "index": [0, 1], "data": [[1, 0.5], [2, 1.5]]}}


def test_rest_response_is_reordered_and_row_counted():
    from smoke_test_endpoint import parse_response

    payload = {"data": [[1, {"p0": 0.2, "p1": 0.8}], [0, {"p0": 0.9, "p1": 0.1}]]}
    assert parse_response(payload, 2, ["p0", "p1"]).tolist() == [[0.9, 0.1], [0.2, 0.8]]
    with pytest.raises(ValueError, match="expected 3 predictions"):
        parse_response(payload, 3, ["p0", "p1"])


def test_rest_outputs_must_match_warehouse_scoring():
    from smoke_test_endpoint import check_outputs

    warehouse = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert check_outputs(warehouse.copy(), warehouse) == []
    assert any("differ" in p for p in check_outputs(np.array([[0.8, 0.2], [0.2, 0.8]]), warehouse))
    assert any("sum to 1" in p for p in check_outputs(np.array([[0.9, 0.9], [0.2, 0.8]]), warehouse))


def test_stale_split_cannot_drop_the_default_versions_service():
    from apply_gateway_config import service_name, stale_split_problem

    live = [service_name("V4"), service_name("V5")]
    stale = [{"service": service_name("V4"), "weight": 100}]
    canary = [{"service": service_name("V4"), "weight": 80}, {"service": service_name("V5"), "weight": 20}]
    assert "stale" in stale_split_problem(stale, "V5", live)
    assert stale_split_problem(canary, "V5", live) is None
    # Rollback sets the default first, so routing to it is allowed
    assert stale_split_problem(stale, "V4", live) is None
    # First deploy: the default version has no service yet
    assert stale_split_problem(stale, "V5", [service_name("V4")]) is None
    assert stale_split_problem(stale, None, live) is None


def test_canary_incumbent_is_the_previous_default_version():
    from apply_gateway_config import service_name
    from deploy_prod_service import pick_incumbent

    v4, v5, v8 = service_name("V4"), service_name("V5"), service_name("V8")
    split = [{"service": v4, "weight": 80}, {"service": v5, "weight": 20}]
    # The champion V8 was gated against keeps the traffic, not the first target
    assert pick_incumbent(split, v8, "V5") == v5
    # Previous default not serving: fall back to the busiest target
    assert pick_incumbent(split, v8, "V3") == v4
    assert pick_incumbent([{"service": v4, "weight": 30}, {"service": v5, "weight": 70}], v8, None) == v5
    assert pick_incumbent([], v8, "V5") is None
    assert pick_incumbent([{"service": v8, "weight": 100}], v8, "V5") is None
