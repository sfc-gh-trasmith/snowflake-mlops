"""PROD service deployment: create, validate, and set canary traffic split.

This script is called by the deploy-prod GitHub Actions workflow.
It deploys a NEW service for the latest model version, validates it,
then updates gateway-config.yml with a canary split (default 80/20).

The traffic-shift workflow reads gateway-config.yml and applies the
ALTER GATEWAY. The engineer later edits the YAML to shift to 100%.

Flow:
  1. Get latest model version from registry
  2. Create versioned service (e.g. MLOPS_FRAUD_DETECTOR_SERVICE_V7)
  3. Wait for READY state
  4. Health check the new service
  5. Update gateway-config.yml with canary split
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import MODEL_NAME
from snowpark_session import create_snowpark_session

PROD_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOW_MLOPS_PROD")
PROD_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
PROD_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "SNOW_MLOPS_PROD_WH")
PROD_COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", "SNOW_MLOPS_PROD_POOL")
GATEWAY_NAME = "FRAUD_DETECTOR_GATEWAY"
SERVICE_PREFIX = f"{MODEL_NAME}_SERVICE"

READY_TIMEOUT_SECONDS = 600  # 10 minutes
READY_POLL_INTERVAL = 15


def get_current_gateway_target(session):
    """Get the current service target from the gateway spec. Returns None if gateway doesn't exist."""
    try:
        rows = session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
    except Exception:
        return None
    if not rows:
        return None
    spec = rows[0]["spec"]
    for line in spec.split("\n"):
        line = line.strip()
        if line.startswith("value:"):
            full_value = line.split("value:")[1].strip()
            service_fqn = full_value.split("!")[0].strip()
            return service_fqn.split(".")[-1]
    return None


def ensure_gateway_exists(session, service_name):
    """Create the gateway if it doesn't exist, pointing to the given service."""
    try:
        session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
        return  # Already exists
    except Exception:
        pass

    print(f"  Creating gateway: {GATEWAY_NAME}")
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{service_name}!inference"
    session.sql(f"""
        CREATE GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}
        FROM SPECIFICATION $$
          spec:
            type: traffic_split
            split_type: custom
            targets:
              - type: endpoint
                value: {fqn}
                weight: 100
        $$
    """).collect()
    print("  Gateway created.")


def wait_for_service_ready(session, service_name, timeout=READY_TIMEOUT_SECONDS):
    """Poll service status until READY or timeout."""
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{service_name}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            status_json = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{fqn}')").collect()[0][0]
            statuses = json.loads(status_json)
            ready = [s for s in statuses if s["status"] == "READY"]
            if ready:
                return True
            print(f"  Status: {statuses[0]['status']} (elapsed: {int(time.time() - start)}s)")
        except Exception as e:
            print(f"  Waiting for service to appear... ({e})")
        time.sleep(READY_POLL_INTERVAL)
    return False


def health_check(session, service_name, model_version_name):
    """Run a test prediction against the new service to validate it works."""
    import numpy as np
    import pandas as pd
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(model_version_name)

    sample = pd.DataFrame(
        [
            {
                "TOTAL_TXN_COUNT": 20,
                "AVG_TXN_AMOUNT": 150.5,
                "MAX_TXN_AMOUNT": 500.0,
                "STDDEV_TXN_AMOUNT": 100.0,
                "UNIQUE_MERCHANTS": 10,
                "ACTIVE_DAYS": 30,
                "LATE_NIGHT_TXN_RATIO": 0.05,
                "CREDIT_SCORE": 700,
                "ACCOUNT_AGE_DAYS": 365,
                "ANNUAL_INCOME": 80000,
            }
        ]
    )

    # Cast sample columns to match model signature so health check doesn't fail on type mismatches
    _NUMPY_DTYPE = {
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "FLOAT": np.float32,
        "DOUBLE": np.float64,
    }
    functions = mv.show_functions()
    predict_proba = [f for f in functions if f["name"].upper() == "PREDICT_PROBA"]
    if predict_proba:
        for feat in predict_proba[0]["signature"].inputs:
            if feat.name in sample.columns:
                np_type = _NUMPY_DTYPE.get(str(feat._dtype).split(".")[-1])
                if np_type:
                    sample[feat.name] = sample[feat.name].astype(np_type)

    result = mv.run(sample, function_name="predict_proba", service_name=service_name)
    # Validate result structure
    assert "output_feature_0" in result.columns, f"Missing output_feature_0 in {result.columns.tolist()}"
    assert "output_feature_1" in result.columns, f"Missing output_feature_1 in {result.columns.tolist()}"
    prob_sum = result["output_feature_0"].iloc[0] + result["output_feature_1"].iloc[0]
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"
    return result


def update_gateway_config(new_service_name, old_service_name):
    """Update gateway-config.yml with a canary split. Does NOT apply — traffic-shift workflow does that."""
    from apply_gateway_config import load_config, update_config_file

    config_path = Path(__file__).resolve().parent.parent / "gateway-config.yml"
    config = load_config(config_path)
    canary_weight = config.get("initial_canary_weight", 20)

    if old_service_name and old_service_name != new_service_name:
        targets = [
            {"service": old_service_name, "weight": 100 - canary_weight},
            {"service": new_service_name, "weight": canary_weight},
        ]
    else:
        targets = [{"service": new_service_name, "weight": 100}]

    update_config_file(config, targets)
    print("  Updated gateway-config.yml:")
    for t in targets:
        print(f"    {t['service']}: {t['weight']}%")
    return targets


def main():
    print("=" * 60)
    print("PROD DEPLOYMENT: Blue/Green with Gateway")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()

    # Step 1: Get the latest model version
    print("\n[1/5] Getting latest model version...")
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {PROD_DATABASE}.{PROD_SCHEMA}")

    default_version = models[0]["default_version_name"]
    print(f"  Model: {MODEL_NAME}, Default version: {default_version}")

    # Step 2: Determine service names
    print("\n[2/5] Determining service names...")
    new_service_name = f"{SERVICE_PREFIX}_{default_version}"
    old_service_name = get_current_gateway_target(session)
    print(f"  New service: {new_service_name}")
    print(f"  Old service: {old_service_name or 'none'}")

    if old_service_name == new_service_name:
        print(f"\n  Service {new_service_name} is already the active gateway target.")
        print("  Nothing to deploy. Exiting.")
        session.close()
        return

    # Step 3: Create the new versioned service
    print(f"\n[3/5] Creating service: {new_service_name}...")
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(default_version)

    try:
        mv.create_service(
            service_name=new_service_name,
            service_compute_pool=PROD_COMPUTE_POOL,
            image_build_compute_pool=PROD_COMPUTE_POOL,
            ingress_enabled=True,
            max_instances=2,
            gpu_requests=None,
            autocapture=True,
        )
        print("  Service creation initiated.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Service already exists - will verify readiness.")
        else:
            raise

    # Step 4: Wait for READY
    print("\n[4/5] Waiting for service to become READY...")
    ready = wait_for_service_ready(session, new_service_name)
    if not ready:
        raise RuntimeError(f"Service {new_service_name} did not become READY within {READY_TIMEOUT_SECONDS}s")
    print("  Service is READY!")

    # Step 5: Health check
    # Health check
    print(f"\n  Running health check against {new_service_name}...")
    result = health_check(session, new_service_name, default_version)
    fraud_prob = result["output_feature_1"].iloc[0]
    print(f"  Health check PASSED (fraud_prob={fraud_prob:.4f})")

    # Step 5: Update gateway config (traffic-shift workflow applies it)
    print("\n[5/5] Updating gateway-config.yml with canary split...")
    targets = update_gateway_config(new_service_name, old_service_name)

    print("\n" + "=" * 60)
    print("PROD SERVICE DEPLOYED (canary)")
    print(f"  Model: {MODEL_NAME} ({default_version})")
    print(f"  Service: {new_service_name}")
    print("  Traffic split:")
    for t in targets:
        print(f"    {t['service']}: {t['weight']}%")
    print("\n  Next: traffic-shift workflow applies the gateway change.")
    print("  To complete cutover: edit gateway-config.yml to 100% and merge.")
    print("=" * 60)
    session.close()


if __name__ == "__main__":
    main()
