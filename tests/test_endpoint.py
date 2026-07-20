"""Integration test for the MLOPS_FRAUD_DETECTOR inference endpoint.

Tests the deployed SPCS inference service via mv.run() (Python SDK).
Validates the service is running, returns valid probability scores,
and handles both legitimate and suspicious transaction patterns.

Usage:
    SNOWFLAKE_CONNECTION_NAME=Trace-CoCo uv run pytest tests/test_endpoint.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import DATABASE, SCHEMA, WAREHOUSE, MODEL_NAME, SERVICE_NAME
from snowpark_session import create_snowpark_session


@pytest.fixture(scope="module")
def session():
    """Create and share a Snowpark session across all tests."""
    sess = create_snowpark_session()
    sess.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def model_version(session):
    """Get model version and ensure service is running."""
    from snowflake.ml.registry import Registry
    import json

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version("V1")

    # Check service status, resume if suspended
    status_json = session.sql(
        f"SELECT SYSTEM$GET_SERVICE_STATUS('{DATABASE}.{SCHEMA}.{SERVICE_NAME}')"
    ).collect()[0][0]
    statuses = json.loads(status_json)
    container_status = statuses[0]["status"]

    if container_status == "SUSPENDED":
        session.sql(f"ALTER SERVICE {DATABASE}.{SCHEMA}.{SERVICE_NAME} RESUME").collect()
        import time
        for _ in range(30):
            time.sleep(10)
            status_json = session.sql(
                f"SELECT SYSTEM$GET_SERVICE_STATUS('{DATABASE}.{SCHEMA}.{SERVICE_NAME}')"
            ).collect()[0][0]
            if '"READY"' in status_json:
                break
        else:
            pytest.fail("Service did not reach READY state within 5 minutes")

    return mv


def _make_sample(
    txn_count=20, avg_amount=150.5, max_amount=500.0, stddev=100.0,
    merchants=10, fraud_count=0, fraud_rate=0.0, active_days=30,
    late_night_ratio=0.05, credit_score=700, account_age=365, income=80000,
):
    """Create a single-row sample DataFrame matching model input schema."""
    import numpy as np
    return pd.DataFrame([{
        "TOTAL_TXN_COUNT": np.int8(txn_count),
        "AVG_TXN_AMOUNT": float(avg_amount),
        "MAX_TXN_AMOUNT": float(max_amount),
        "STDDEV_TXN_AMOUNT": float(stddev),
        "UNIQUE_MERCHANTS": np.int8(merchants),
        "HISTORICAL_FRAUD_COUNT": np.int8(fraud_count),
        "HISTORICAL_FRAUD_RATE": float(fraud_rate),
        "ACTIVE_DAYS": np.int8(active_days),
        "LATE_NIGHT_TXN_RATIO": float(late_night_ratio),
        "CREDIT_SCORE": np.int16(credit_score),
        "ACCOUNT_AGE_DAYS": np.int16(account_age),
        "ANNUAL_INCOME": np.int32(income),
    }])


class TestServiceHealth:
    """Test that the inference service is responsive."""

    def test_service_is_running(self, session):
        """Service should be in READY state."""
        import json
        status_json = session.sql(
            f"SELECT SYSTEM$GET_SERVICE_STATUS('{DATABASE}.{SCHEMA}.{SERVICE_NAME}')"
        ).collect()[0][0]
        statuses = json.loads(status_json)
        ready_containers = [s for s in statuses if s["status"] == "READY"]
        assert len(ready_containers) > 0, f"No READY containers: {statuses}"

    def test_endpoint_exists(self, session):
        """Service should have a public ingress endpoint."""
        rows = session.sql(
            f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{SERVICE_NAME}"
        ).collect()
        assert len(rows) > 0, "No endpoints found"
        assert str(rows[0]["is_public"]).lower() == "true", "Endpoint is not public (ingress not enabled)"


class TestPredictions:
    """Test model predictions via the service."""

    def test_predict_proba_returns_two_columns(self, model_version):
        """predict_proba should return probabilities for both classes."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=SERVICE_NAME)
        assert "output_feature_0" in result.columns
        assert "output_feature_1" in result.columns

    def test_probabilities_sum_to_one(self, model_version):
        """Class probabilities should sum to approximately 1.0."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=SERVICE_NAME)
        prob_sum = result["output_feature_0"].iloc[0] + result["output_feature_1"].iloc[0]
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"

    def test_probabilities_in_valid_range(self, model_version):
        """All probabilities should be between 0 and 1."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=SERVICE_NAME)
        for col in ["output_feature_0", "output_feature_1"]:
            val = result[col].iloc[0]
            assert 0.0 <= val <= 1.0, f"{col} = {val} is outside [0, 1]"

    def test_predict_returns_binary(self, model_version):
        """predict should return 0 or 1."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict", service_name=SERVICE_NAME)
        pred = result["output_feature_0"].iloc[0]
        assert pred in (0, 1), f"Prediction {pred} is not binary"

    def test_batch_prediction(self, model_version):
        """Service should handle multiple rows in a single request."""
        batch = pd.concat([
            _make_sample(),
            _make_sample(txn_count=3, avg_amount=2000, fraud_rate=0.8, credit_score=350),
            _make_sample(txn_count=50, avg_amount=25, fraud_rate=0.0, credit_score=800),
        ], ignore_index=True)
        result = model_version.run(batch, function_name="predict_proba", service_name=SERVICE_NAME)
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"

    def test_high_risk_gets_higher_fraud_probability(self, model_version):
        """A suspicious profile should get higher fraud probability than a clean one."""
        clean = _make_sample(fraud_count=0, fraud_rate=0.0, credit_score=800, late_night_ratio=0.0)
        suspicious = _make_sample(
            txn_count=3, avg_amount=3000, max_amount=10000, fraud_count=5,
            fraud_rate=0.8, active_days=5, late_night_ratio=0.9, credit_score=350,
            account_age=30, income=20000,
        )

        clean_result = model_version.run(clean, function_name="predict_proba", service_name=SERVICE_NAME)
        sus_result = model_version.run(suspicious, function_name="predict_proba", service_name=SERVICE_NAME)

        clean_fraud_prob = clean_result["output_feature_1"].iloc[0]
        sus_fraud_prob = sus_result["output_feature_1"].iloc[0]
        assert sus_fraud_prob > clean_fraud_prob, (
            f"Suspicious ({sus_fraud_prob:.4f}) should have higher fraud prob than clean ({clean_fraud_prob:.4f})"
        )
