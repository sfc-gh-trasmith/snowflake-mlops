"""ML Pipeline DAG - Snowflake Task Graph Orchestration.

Orchestrates the ML workflow as a Snowflake Task Graph:
  REFRESH_FEATURES -> TRAIN_MODEL -> REPLICATE_MODEL (STAGE only)

The DAG is deployed from git (via CI/CD or manually) and executed via
EXECUTE TASK. Config is passed at runtime via USING CONFIG.

Environments:
  - DEV: Deploy + run manually for experimentation
  - STAGE: Deployed by CI (deploy.yml), executed on every merge
  - PROD: No DAG (serving only, no training)

Usage:
    python source/pipeline/ml_pipeline_dag.py --deploy --env dev
    python source/pipeline/ml_pipeline_dag.py --deploy --env stage
    python source/pipeline/ml_pipeline_dag.py --run --env stage
    python source/pipeline/ml_pipeline_dag.py --status --env dev
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MIN_AUC_ROC,
    MIN_PRECISION,
    MIN_RECALL,
)
from snowpark_session import create_snowpark_session


# Environment configurations
ENV_CONFIG = {
    "dev": {
        "database": "SNOW_MLOPS_DEV",
        "schema": "ML",
        "warehouse": "SNOW_MLOPS_DEV_WH",
        "compute_pool": "SNOW_MLOPS_DEV_POOL",
        "source_database": "SNOW_MLOPS_PROD",
        "source_schema": "ML",
    },
    "stage": {
        "database": "SNOW_MLOPS_STAGE",
        "schema": "ML",
        "warehouse": "SNOW_MLOPS_STAGE_WH",
        "compute_pool": "SNOW_MLOPS_STAGE_POOL",
        "source_database": "SNOW_MLOPS_PROD",
        "source_schema": "ML",
    },
}

DAG_NAME = "ML_TRAINING_PIPELINE"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"


def get_env_config(env: str) -> dict:
    """Get configuration for the specified environment."""
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Use 'dev' or 'stage'.")
    return ENV_CONFIG[env]


def deploy_dag(env: str):
    """Deploy (CREATE OR REPLACE) the Task DAG for the specified environment."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    pool = cfg["compute_pool"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    fv_table = f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Deploying Task DAG: {db}.{schema}.{DAG_NAME}")
    print(f"  Environment: {env.upper()}")
    print(f"  Compute Pool: {pool}")
    print(f"  Feature View: {fv_table}")

    # Create alerts table
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.PIPELINE_ALERTS (
            ALERT_TIME TIMESTAMP_NTZ,
            MESSAGE VARCHAR,
            METRICS VARIANT
        )
    """).collect()

    # Create the root task: TRAIN_AND_REGISTER
    # This task submits the @remote training job and waits for it
    train_sql = f"""
BEGIN
    LET feature_table VARCHAR := '{fv_table}';
    LET src_database VARCHAR := '{src_db}';
    LET src_schema VARCHAR := '{src_schema}';
    LET target_db VARCHAR := '{db}';
    LET target_schema VARCHAR := '{schema}';
    LET model_name VARCHAR := '{MODEL_NAME}';

    -- Auto-increment version
    LET version_name VARCHAR := 'V1';
    BEGIN
        LET versions RESULTSET := (SHOW VERSIONS IN MODEL IDENTIFIER(:target_db || '.' || :target_schema || '.' || :model_name));
        LET max_v NUMBER := (SELECT MAX(TRY_TO_NUMBER(REPLACE("name", 'V', ''))) FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())));
        version_name := 'V' || (:max_v + 1)::VARCHAR;
    EXCEPTION
        WHEN OTHER THEN
            version_name := 'V1';
    END;

    -- Store version for downstream tasks
    CALL SYSTEM$SET_RETURN_VALUE(:version_name);
END;
"""

    # Root task: orchestrates the pipeline
    session.sql(f"""
        CREATE OR REPLACE TASK {db}.{schema}.{DAG_NAME}
            WAREHOUSE = {wh}
        AS
        {train_sql}
    """).collect()

    # Child task: Execute the actual training via stored procedure
    # The training itself is done by run_stage_pipeline.py / run_training_job.py
    # which the CI workflow calls after deploying the DAG
    # For now, the DAG is a placeholder for future scheduled runs

    print(f"  Task created: {db}.{schema}.{DAG_NAME}")

    # Resume the task (needed for EXECUTE TASK to work)
    session.sql(f"ALTER TASK {db}.{schema}.{DAG_NAME} RESUME").collect()
    print("  Task resumed (ready for EXECUTE TASK)")

    session.close()
    print("\nDeploy complete.")


def run_dag(env: str, config_override: dict = None):
    """Trigger the Task DAG with runtime config."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    runtime_config = {
        "environment": env,
        "feature_view": f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}",
        "min_auc_roc": str(MIN_AUC_ROC),
        "min_precision": str(MIN_PRECISION),
        "min_recall": str(MIN_RECALL),
        **(config_override or {}),
    }

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    config_json = json.dumps(runtime_config)
    print(f"Executing Task: {db}.{schema}.{DAG_NAME}")
    print(f"  Config: {config_json}")

    session.sql(f"""
        EXECUTE TASK {db}.{schema}.{DAG_NAME}
        USING CONFIG = $${config_json}$$
    """).collect()

    print("  Task triggered! Monitor with --status")
    session.close()


def show_status(env: str):
    """Show recent task execution history."""
    cfg = get_env_config(env)
    db = cfg["database"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Task history for {db} (last 24h):\n")
    rows = session.sql(f"""
        SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME, RETURN_VALUE, ERROR_MESSAGE
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
            SCHEDULED_TIME_RANGE_START => DATEADD('hour', -24, CURRENT_TIMESTAMP()),
            RESULT_LIMIT => 20
        ))
        WHERE DATABASE_NAME = '{db}'
        ORDER BY SCHEDULED_TIME DESC
    """).collect()

    if not rows:
        print("  No task runs found in the last 24 hours.")
    else:
        for row in rows:
            ts = str(row["SCHEDULED_TIME"])[:19]
            state = row["STATE"]
            name = row["NAME"]
            ret = row["RETURN_VALUE"] or ""
            err = row["ERROR_MESSAGE"] or ""
            print(f"  [{ts}] {name}: {state} {ret[:80]} {err[:80]}")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Task DAG")
    parser.add_argument("--deploy", action="store_true", help="Deploy (create/replace) the Task DAG")
    parser.add_argument("--run", action="store_true", help="Trigger a pipeline execution")
    parser.add_argument("--status", action="store_true", help="Show recent task history")
    parser.add_argument(
        "--env",
        default=os.getenv("ML_ENV", "dev"),
        choices=["dev", "stage"],
        help="Target environment (default: $ML_ENV or 'dev')",
    )
    args = parser.parse_args()

    if args.deploy:
        deploy_dag(args.env)
    elif args.run:
        run_dag(args.env)
    elif args.status:
        show_status(args.env)
    else:
        parser.print_help()
