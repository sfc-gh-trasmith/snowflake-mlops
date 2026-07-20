"""Deploy PROD inference service from the replicated model.

This script is called by the deploy-prod.yml GitHub Actions workflow.
It deploys the MLOPS_FRAUD_DETECTOR model (replicated from STAGE)
as a SPCS inference service with REST gateway in PROD.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"
PROD_WAREHOUSE = "SNOW_MLOPS_PROD_WH"
PROD_COMPUTE_POOL = "SNOW_MLOPS_PROD_POOL"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"
SERVICE_NAME = "MLOPS_FRAUD_DETECTOR_SERVICE"


def main():
    print("=" * 60)
    print("PROD DEPLOYMENT: Inference Service")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()

    # Verify model exists in PROD
    print(f"\n[1/3] Verifying model exists: {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}")
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {PROD_DATABASE}.{PROD_SCHEMA}. Run STAGE pipeline first.")
    print(f"  Found: {MODEL_NAME} (versions: {models[0]['versions']})")

    # Deploy inference service using default (latest) version
    print(f"\n[2/3] Deploying inference service: {SERVICE_NAME}")
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)

    # Use the default version (set during replication)
    default_version = models[0]["default_version_name"]
    print(f"  Using default version: {default_version}")
    mv = model.version(default_version)

    try:
        mv.create_service(
            service_name=SERVICE_NAME,
            service_compute_pool=PROD_COMPUTE_POOL,
            image_build_compute_pool=PROD_COMPUTE_POOL,
            ingress_enabled=True,
            max_instances=2,
            gpu_requests=None,
        )
        print("  Service created successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Service already exists - skipping creation")
        else:
            raise

    # Verify service is running
    print("\n[3/3] Checking service status...")
    try:
        status_json = session.sql(
            f"SELECT SYSTEM$GET_SERVICE_STATUS('{PROD_DATABASE}.{PROD_SCHEMA}.{SERVICE_NAME}')"
        ).collect()[0][0]
        statuses = json.loads(status_json)
        ready = [s for s in statuses if s["status"] == "READY"]
        print(f"  Service status: {len(ready)}/{len(statuses)} containers READY")

        # Show endpoint
        endpoints = session.sql(f"SHOW ENDPOINTS IN SERVICE {PROD_DATABASE}.{PROD_SCHEMA}.{SERVICE_NAME}").collect()
        if endpoints:
            print(f"  REST Gateway: https://{endpoints[0]['ingress_url']}")
    except Exception as e:
        print(f"  Service still starting (this is normal): {e}")
        print(f"  Check status: SELECT SYSTEM$GET_SERVICE_STATUS('{PROD_DATABASE}.{PROD_SCHEMA}.{SERVICE_NAME}')")

    print("\n" + "=" * 60)
    print("PROD DEPLOYMENT COMPLETE")
    print("=" * 60)
    session.close()


if __name__ == "__main__":
    main()
