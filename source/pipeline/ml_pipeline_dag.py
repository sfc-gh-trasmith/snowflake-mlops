"""ML Pipeline Task DAG — Deploy and manage the Snowflake Task graph.

Creates a 3-task dependency chain:
  ML_PIPELINE_FEATURE_ENG → ML_PIPELINE_TRAIN → ML_PIPELINE_EVALUATE

Each task runs as an ML Job. Compute mode (warehouse or SPCS) is configurable
per step via PIPELINE_CONFIG in source/config.py.

Usage:
    python source/pipeline/ml_pipeline_dag.py --deploy --env stage
    python source/pipeline/ml_pipeline_dag.py --run --env stage
    python source/pipeline/ml_pipeline_dag.py --status --env stage
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
    PIPELINE_CONFIG,
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

DAG_ROOT = "ML_TRAINING_PIPELINE"
TASK_FEATURE_ENG = "ML_PIPELINE_FEATURE_ENG"
TASK_TRAIN = "ML_PIPELINE_TRAIN"
TASK_EVALUATE = "ML_PIPELINE_EVALUATE"


def get_env_config(env: str) -> dict:
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Use 'dev' or 'stage'.")
    return ENV_CONFIG[env]


def deploy_dag(env: str):
    """Deploy the 3-task ML pipeline DAG."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    pool = cfg["compute_pool"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    fv = f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"
    timeout = PIPELINE_CONFIG.get("task_timeout_ms", "7200000")

    fe_compute = PIPELINE_CONFIG.get("feature_engineering_compute", "warehouse")
    train_compute = PIPELINE_CONFIG.get("training_compute", "spcs")

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Deploying Task DAG: {db}.{schema}.{DAG_ROOT}")
    print(f"  Environment: {env.upper()}")
    print(f"  Feature eng compute: {fe_compute}")
    print(f"  Training compute: {train_compute}")
    print(f"  Compute Pool: {pool}")

    # Create results table
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.PIPELINE_RESULTS (
            STEP VARCHAR,
            STATUS VARCHAR,
            RESULT VARIANT,
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()

    # Create pipeline stage for code + artifacts
    session.sql(f"CREATE STAGE IF NOT EXISTS {db}.{schema}.PIPELINE_STAGE").collect()

    # --- Task 1: Feature Engineering (root task) ---
    if fe_compute == "warehouse":
        # For warehouse mode, feature engineering is handled by the GH Actions upload step
        # or inline SQL. The task just marks completion.
        session.sql(f"""
            CREATE OR REPLACE TASK {db}.{schema}.{TASK_FEATURE_ENG}
                WAREHOUSE = {wh}
                USER_TASK_TIMEOUT_MS = {timeout}
            AS
            BEGIN
                -- Feature Views are registered via the uploaded pipeline code
                -- This task triggers the ML Job for feature engineering
                LET job_id VARCHAR;
                CALL SYSTEM$ML_JOB_SUBMIT(
                    '{pool}',
                    '@{db}.{schema}.PIPELINE_STAGE/source/pipeline/tasks/feature_engineering.py',
                    '{{"pip_requirements": ["snowflake-ml-python"], "env_vars": {{"PIPELINE_DATABASE": "{db}", "PIPELINE_SCHEMA": "{schema}", "PIPELINE_WAREHOUSE": "{wh}"}}}}'
                );
                CALL SYSTEM$SET_RETURN_VALUE('feature_engineering_submitted');
            END;
        """).collect()
    else:
        session.sql(f"""
            CREATE OR REPLACE TASK {db}.{schema}.{TASK_FEATURE_ENG}
                WAREHOUSE = {wh}
                USER_TASK_TIMEOUT_MS = {timeout}
            AS
            BEGIN
                CALL SYSTEM$ML_JOB_SUBMIT(
                    '{pool}',
                    '@{db}.{schema}.PIPELINE_STAGE/source/pipeline/tasks/feature_engineering.py',
                    '{{"pip_requirements": ["snowflake-ml-python"], "env_vars": {{"PIPELINE_DATABASE": "{db}", "PIPELINE_SCHEMA": "{schema}", "PIPELINE_WAREHOUSE": "{wh}"}}}}'
                );
                CALL SYSTEM$SET_RETURN_VALUE('feature_engineering_submitted');
            END;
        """).collect()
    print(f"  Created: {TASK_FEATURE_ENG} (compute: {fe_compute})")

    # --- Task 2: Model Training ---
    train_env_vars = json.dumps(
        {
            "PIPELINE_DATABASE": db,
            "PIPELINE_SCHEMA": schema,
            "PIPELINE_WAREHOUSE": wh,
            "PIPELINE_SOURCE_DATABASE": src_db,
            "PIPELINE_SOURCE_SCHEMA": src_schema,
            "PIPELINE_FEATURE_VIEW": fv,
            "PIPELINE_N_ESTIMATORS": PIPELINE_CONFIG.get("n_estimators", "200"),
            "PIPELINE_LEARNING_RATE": PIPELINE_CONFIG.get("learning_rate", "0.1"),
            "PIPELINE_MAX_DEPTH": PIPELINE_CONFIG.get("max_depth", "6"),
            "PIPELINE_SCALE_POS_WEIGHT": PIPELINE_CONFIG.get("scale_pos_weight", "33"),
        }
    ).replace("'", "''")

    session.sql(f"""
        CREATE OR REPLACE TASK {db}.{schema}.{TASK_TRAIN}
            WAREHOUSE = {wh}
            USER_TASK_TIMEOUT_MS = {timeout}
            AFTER {db}.{schema}.{TASK_FEATURE_ENG}
        AS
        BEGIN
            CALL SYSTEM$ML_JOB_SUBMIT(
                '{pool}',
                '@{db}.{schema}.PIPELINE_STAGE/source/pipeline/tasks/train_model.py',
                '{{"pip_requirements": ["xgboost", "scikit-learn", "snowflake-ml-python"], "env_vars": {train_env_vars}}}'
            );
            CALL SYSTEM$SET_RETURN_VALUE('training_submitted');
        END;
    """).collect()
    print(f"  Created: {TASK_TRAIN} (compute: spcs/{pool})")

    # --- Task 3: Evaluation ---
    eval_env_vars = json.dumps(
        {
            "PIPELINE_DATABASE": db,
            "PIPELINE_SCHEMA": schema,
            "PIPELINE_WAREHOUSE": wh,
        }
    ).replace("'", "''")

    session.sql(f"""
        CREATE OR REPLACE TASK {db}.{schema}.{TASK_EVALUATE}
            WAREHOUSE = {wh}
            USER_TASK_TIMEOUT_MS = {timeout}
            AFTER {db}.{schema}.{TASK_TRAIN}
        AS
        BEGIN
            CALL SYSTEM$ML_JOB_SUBMIT(
                '{pool}',
                '@{db}.{schema}.PIPELINE_STAGE/source/pipeline/tasks/evaluate_model.py',
                '{{"pip_requirements": ["snowflake-ml-python"], "env_vars": {eval_env_vars}}}'
            );
            CALL SYSTEM$SET_RETURN_VALUE('evaluation_submitted');
        END;
    """).collect()
    print(f"  Created: {TASK_EVALUATE} (compute: spcs/{pool})")

    # Resume child tasks only (root task is triggered via EXECUTE TASK, not scheduled)
    for task in [TASK_TRAIN, TASK_EVALUATE]:
        session.sql(f"ALTER TASK {db}.{schema}.{task} RESUME").collect()
    print("  Child tasks resumed (root task triggered on-demand via EXECUTE TASK).")

    session.close()
    print("\nDAG deployment complete.")


def run_dag(env: str):
    """Trigger the Task DAG."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    # Clear previous results
    session.sql(f"DELETE FROM {db}.{schema}.PIPELINE_RESULTS").collect()

    print(f"Executing Task: {db}.{schema}.{TASK_FEATURE_ENG}")
    session.sql(f"EXECUTE TASK {db}.{schema}.{TASK_FEATURE_ENG}").collect()
    print("  Task triggered!")
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
    parser.add_argument("--deploy", action="store_true", help="Deploy the Task DAG")
    parser.add_argument("--run", action="store_true", help="Trigger pipeline execution")
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
