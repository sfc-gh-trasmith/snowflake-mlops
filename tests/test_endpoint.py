"""Live integration tests for the PROD real-time endpoint over HTTPS REST.

The same checks the deploy-prod smoke test runs (scripts/smoke_test_endpoint.py),
as pytest cases you can run against PROD from a laptop. Inputs are real rows
from the batch input table and expectations come from the model signature, so
nothing here is specific to the example model.

Not part of the unit suite (CI runs it as the post-deploy smoke test step).

Usage:
    SNOWFLAKE_DATABASE=SNOW_MLOPS_PROD SNOWFLAKE_WAREHOUSE=SNOW_MLOPS_PROD_WH \\
        uv run pytest tests/test_endpoint.py -v

Auth: SNOWFLAKE_PAT if set, otherwise a temporary 1-day PAT is issued for the
connected user and removed afterwards.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test_endpoint as smoke
from apply_gateway_config import GATEWAY_NAME, get_current_gateway_targets
from config import MODEL_NAME, PREDICT_FUNCTION, TASK_TYPE, batch_input_table
from serving.signature import cast_pandas, input_features, output_columns, run_outputs, validate_outputs
from snowpark_session import create_snowpark_session

DB, SCHEMA = smoke.PROD_DATABASE, smoke.PROD_SCHEMA


@pytest.fixture(scope="module")
def session():
    sess = create_snowpark_session()
    sess.sql(f"USE WAREHOUSE {smoke.PROD_WAREHOUSE}").collect()
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def targets(session):
    targets = get_current_gateway_targets(session, DB, SCHEMA, GATEWAY_NAME)
    if not targets:
        pytest.skip(f"No gateway {DB}.{SCHEMA}.{GATEWAY_NAME} (realtime not deployed)")
    return targets


@pytest.fixture(scope="module")
def model(session):
    from snowflake.ml.registry import Registry

    return Registry(session=session, database_name=DB, schema_name=SCHEMA).get_model(MODEL_NAME)


@pytest.fixture(scope="module")
def token(session):
    with smoke.EphemeralPAT(session) as tok:
        yield tok


@pytest.fixture(scope="module")
def sample(session, model, targets):
    mv = model.version(targets[0]["service"].rsplit("_SERVICE_", 1)[-1])
    feats = input_features(mv, PREDICT_FUNCTION)
    df = session.table(batch_input_table(DB, SCHEMA)).select(*feats).limit(smoke.SMOKE_ROWS).to_pandas()
    assert not df.empty, "batch input table is empty"
    return df


def _url(session, sql, where=None):
    return f"https://{smoke._ingress_host(session, sql, 'ingress_url', where)}{smoke.method_path(PREDICT_FUNCTION)}"


def _service_url(session, service):
    return _url(session, f"SHOW ENDPOINTS IN SERVICE {DB}.{SCHEMA}.{service}", lambda r: r["name"] == "inference")


def test_gateway_weights_sum_to_100(targets):
    assert sum(t["weight"] for t in targets) == 100


def test_every_target_service_is_ready(session, targets):
    import json

    for t in targets:
        status = json.loads(
            session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{DB}.{SCHEMA}.{t['service']}')").collect()[0][0]
        )
        assert any(s["status"] == "READY" for s in status), f"{t['service']}: {status}"


def test_only_gateway_targets_are_running(session, targets):
    """Services outside the split only cost compute; apply_gateway_config drops them."""
    from apply_gateway_config import model_services, unused_services

    assert unused_services(model_services(session, DB, SCHEMA), targets) == []


def test_gateway_rest_predictions_are_valid(session, model, targets, token, sample):
    mv = model.version(targets[0]["service"].rsplit("_SERVICE_", 1)[-1])
    url = _url(session, f"DESC GATEWAY {DB}.{SCHEMA}.{GATEWAY_NAME}")
    out, _ = smoke.post(url, token, cast_pandas(sample, mv, PREDICT_FUNCTION), output_columns(mv, PREDICT_FUNCTION))
    assert out.shape[0] == len(sample)
    assert validate_outputs(TASK_TYPE, out) == []


def test_each_service_matches_warehouse_scoring(session, model, targets, token, sample):
    for t in targets:
        mv = model.version(t["service"].rsplit("_SERVICE_", 1)[-1])
        X = cast_pandas(sample, mv, PREDICT_FUNCTION)
        out, _ = smoke.post(_service_url(session, t["service"]), token, X, output_columns(mv, PREDICT_FUNCTION))
        assert smoke.check_outputs(out, run_outputs(mv, sample, PREDICT_FUNCTION)) == [], t["service"]


def test_single_row_request(session, model, targets, token, sample):
    """Online use case: one entity per request."""
    mv = model.version(targets[0]["service"].rsplit("_SERVICE_", 1)[-1])
    X = cast_pandas(sample.head(1), mv, PREDICT_FUNCTION)
    out, latency_ms = smoke.post(
        _service_url(session, targets[0]["service"]), token, X, output_columns(mv, PREDICT_FUNCTION)
    )
    assert out.shape[0] == 1 and np.isfinite(out).all()
    print(f"single-row latency: {latency_ms:.0f} ms")


def test_invalid_token_is_rejected(session, targets, sample):
    with pytest.raises(RuntimeError, match="HTTP 40"):
        smoke.post(_service_url(session, targets[0]["service"]), "not-a-token", sample.head(1), ["x"])
