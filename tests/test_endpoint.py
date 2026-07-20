"""Integration tests for the MLOPS_FRAUD_DETECTOR inference endpoint.

Tests both the deployed SPCS inference service and the Snowflake Gateway
(stable production URL). Validates predictions, probability ranges,
and the gateway routing layer.

Usage:
    SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run pytest tests/test_endpoint.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"
PROD_WAREHOUSE = "SNOW_MLOPS_PROD_WH"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"
GATEWAY_NAME = "FRAUD_DETECTOR_GATEWAY"


@pytest.fixture(scope="module")
def session():
    """Create and share a Snowpark session across all tests."""
    sess = create_snowpark_session()
    sess.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def model_version(session):
    """Get the current default model version."""
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)

    # Use the default (latest) version
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    default_version = models[0]["default_version_name"]
    return model.version(default_version)


@pytest.fixture(scope="module")
def active_service_name(session):
    """Get the service currently targeted by the gateway."""
    rows = session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
    spec = rows[0]["spec"]
    for line in spec.split("\n"):
        line = line.strip()
        if line.startswith("value:"):
            full_value = line.split("value:")[1].strip()
            service_fqn = full_value.split("!")[0].strip()
            return service_fqn.split(".")[-1]
    pytest.fail("Could not determine active service from gateway spec")


def _make_sample(
    txn_count=20,
    avg_amount=150.5,
    max_amount=500.0,
    stddev=100.0,
    merchants=10,
    fraud_count=0,
    fraud_rate=0.0,
    active_days=30,
    late_night_ratio=0.05,
    credit_score=700,
    account_age=365,
    income=80000,
):
    """Create a single-row sample DataFrame matching model input schema."""
    return pd.DataFrame(
        [
            {
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
            }
        ]
    )


class TestGatewayHealth:
    """Test that the gateway and underlying service are operational."""

    def test_gateway_exists(self, session):
        """Gateway should exist and have an ingress URL."""
        rows = session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
        assert len(rows) > 0, "Gateway not found"
        url = rows[0]["ingress_url"]
        assert url and "provisioning" not in url.lower(), f"Gateway not ready: {url}"

    def test_active_service_is_running(self, session, active_service_name):
        """The service behind the gateway should be READY."""
        status_json = session.sql(
            f"SELECT SYSTEM$GET_SERVICE_STATUS('{PROD_DATABASE}.{PROD_SCHEMA}.{active_service_name}')"
        ).collect()[0][0]
        statuses = json.loads(status_json)
        ready = [s for s in statuses if s["status"] == "READY"]
        assert len(ready) > 0, f"No READY containers: {statuses}"

    def test_service_has_public_endpoint(self, session, active_service_name):
        """Active service should have ingress enabled."""
        rows = session.sql(f"SHOW ENDPOINTS IN SERVICE {PROD_DATABASE}.{PROD_SCHEMA}.{active_service_name}").collect()
        assert len(rows) > 0, "No endpoints found"
        assert str(rows[0]["is_public"]).lower() == "true"


class TestPredictions:
    """Test model predictions via the service behind the gateway."""

    def test_predict_proba_returns_two_columns(self, model_version, active_service_name):
        """predict_proba should return probabilities for both classes."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=active_service_name)
        assert "output_feature_0" in result.columns
        assert "output_feature_1" in result.columns

    def test_probabilities_sum_to_one(self, model_version, active_service_name):
        """Class probabilities should sum to approximately 1.0."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=active_service_name)
        prob_sum = result["output_feature_0"].iloc[0] + result["output_feature_1"].iloc[0]
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"

    def test_probabilities_in_valid_range(self, model_version, active_service_name):
        """All probabilities should be between 0 and 1."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict_proba", service_name=active_service_name)
        for col in ["output_feature_0", "output_feature_1"]:
            val = result[col].iloc[0]
            assert 0.0 <= val <= 1.0, f"{col} = {val} is outside [0, 1]"

    def test_predict_returns_binary(self, model_version, active_service_name):
        """predict should return 0 or 1."""
        sample = _make_sample()
        result = model_version.run(sample, function_name="predict", service_name=active_service_name)
        pred = result["output_feature_0"].iloc[0]
        assert pred in (0, 1), f"Prediction {pred} is not binary"

    def test_batch_prediction(self, model_version, active_service_name):
        """Service should handle multiple rows in a single request."""
        batch = pd.concat(
            [
                _make_sample(),
                _make_sample(txn_count=3, avg_amount=2000, fraud_rate=0.8, credit_score=350),
                _make_sample(txn_count=50, avg_amount=25, fraud_rate=0.0, credit_score=800),
            ],
            ignore_index=True,
        )
        result = model_version.run(batch, function_name="predict_proba", service_name=active_service_name)
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"

    def test_high_risk_gets_higher_fraud_probability(self, model_version, active_service_name):
        """A suspicious profile should get higher fraud probability than a clean one."""
        clean = _make_sample(fraud_count=0, fraud_rate=0.0, credit_score=800, late_night_ratio=0.0)
        suspicious = _make_sample(
            txn_count=3,
            avg_amount=3000,
            max_amount=10000,
            fraud_count=5,
            fraud_rate=0.8,
            active_days=5,
            late_night_ratio=0.9,
            credit_score=350,
            account_age=30,
            income=20000,
        )

        clean_result = model_version.run(clean, function_name="predict_proba", service_name=active_service_name)
        sus_result = model_version.run(suspicious, function_name="predict_proba", service_name=active_service_name)

        clean_fraud_prob = clean_result["output_feature_1"].iloc[0]
        sus_fraud_prob = sus_result["output_feature_1"].iloc[0]
        assert sus_fraud_prob > clean_fraud_prob, (
            f"Suspicious ({sus_fraud_prob:.4f}) should have higher fraud prob than clean ({clean_fraud_prob:.4f})"
        )
