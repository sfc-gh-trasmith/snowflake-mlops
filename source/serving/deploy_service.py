"""Deploy model to SPCS inference service with REST gateway."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE, SCHEMA, WAREHOUSE, COMPUTE_POOL, MODEL_NAME, SERVICE_NAME
from snowpark_session import create_snowpark_session


def deploy_inference_service(
    version_name: str = "V1",
    max_instances: int = 2,
    session=None,
) -> object:
    """Deploy model version to SPCS for real-time inference.

    Creates a containerized inference service with:
    - REST gateway (ingress_enabled=True) for external access
    - Auto-scaling up to max_instances
    - CPU-based inference (no GPU needed for XGBoost)

    Args:
        version_name: Model version to deploy
        max_instances: Max horizontal scale
        session: Optional Snowpark session

    Returns:
        ModelVersion object with active service
    """
    from snowflake.ml.registry import Registry

    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(version_name)

    print(f"Deploying {MODEL_NAME}/{version_name} to SPCS...")
    print(f"  Compute Pool: {COMPUTE_POOL}")
    print(f"  Max Instances: {max_instances}")
    print("  Ingress: Enabled (REST gateway)")

    mv.create_service(
        service_name=SERVICE_NAME,
        service_compute_pool=COMPUTE_POOL,
        image_build_compute_pool=COMPUTE_POOL,
        ingress_enabled=True,
        max_instances=max_instances,
        gpu_requests=None,
    )

    print(f"\nService deployed: {DATABASE}.{SCHEMA}.{SERVICE_NAME}")
    print(f"  Status: Check with DESCRIBE SERVICE {DATABASE}.{SCHEMA}.{SERVICE_NAME}")
    print(f"  Test SQL: SELECT {DATABASE}.{SCHEMA}.{MODEL_NAME}!PREDICT(...)")

    if close_session:
        session.close()

    return mv


def get_service_status(session=None) -> dict:
    """Check inference service status."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    result = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{DATABASE}.{SCHEMA}.{SERVICE_NAME}')").collect()

    if close_session:
        session.close()

    import json

    status = json.loads(result[0][0])
    return status


def test_inference(session=None, sample_data=None):
    """Test the deployed service with sample data."""
    from snowflake.ml.registry import Registry

    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.default

    if sample_data is None:
        # Get a few rows from the training data
        sample_data = session.table(f"{DATABASE}.{SCHEMA}.RAW_TRANSACTIONS").limit(5).to_pandas()

    predictions = mv.run(sample_data, function_name="predict_proba")
    print("Inference results:")
    print(predictions)

    if close_session:
        session.close()

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Deploy service")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--test", action="store_true", help="Test inference")
    args = parser.parse_args()

    if args.deploy:
        deploy_inference_service()
    elif args.status:
        import json

        print(json.dumps(get_service_status(), indent=2))
    elif args.test:
        test_inference()
    else:
        print("Usage: --deploy | --status | --test")
