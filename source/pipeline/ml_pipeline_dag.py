"""ML Pipeline Task DAG — Python SDK with configurable compute per step.

Uses snowflake.core.task.dagv1 (DAG, DAGTask, DAGOperation) for task management
and snowflake.ml.jobs.remote for ML Job definitions on compute pools.

Each step can run on either:
  - Warehouse: StoredProcedureCall (for SQL-heavy / lightweight Python)
  - SPCS Compute Pool: @remote ML Job (for training, GPU, custom packages)

Configured via PIPELINE_CONFIG in source/config.py:
  "feature_engineering_compute": "warehouse" | "spcs"
  "training_compute": "warehouse" | "spcs"
  "evaluation_compute": "warehouse" | "spcs"

Usage:
    python source/pipeline/ml_pipeline_dag.py --deploy --env stage
    python source/pipeline/ml_pipeline_dag.py --execute --env stage
    python source/pipeline/ml_pipeline_dag.py --status --env stage
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EXPERIMENT_CONFIG,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    ML_RUNTIME_DEPS,
    PIPELINE_CONFIG,
    RETRAIN_CONFIG,
    TRAINING_PARAMS,
)
from snowflake.core import Root
from snowflake.core._common import CreateMode
from snowflake.core.task import StoredProcedureCall
from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask
from snowflake.ml.jobs import remote
from snowflake.snowpark import Session
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


def get_env_config(env: str) -> dict:
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Use 'dev' or 'stage'.")
    return ENV_CONFIG[env]


# ─── ML Job Definitions (@remote) ────────────────────────────────────────────
# These run on compute pools when the corresponding config is set to "spcs"


def build_feature_eng_remote(cfg: dict):
    """Build the @remote-decorated feature engineering function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    stage = f"@{db}.{schema}.PIPELINE_STAGE"

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=["snowflake-ml-python"],
    )
    def feature_engineering() -> str:
        import json

        from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
        from snowflake.snowpark import Session as _Session
        from snowflake.snowpark import functions as F

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        fs = FeatureStore(
            session=session,
            database=db,
            name=schema,
            default_warehouse=wh,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

        # Register entities
        customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
        fs.register_entity(customer_entity)

        # Build and register CUSTOMER_RISK_FEATURES
        txn = session.table(f"{src_db}.{src_schema}.RAW_TRANSACTIONS")
        cust = session.table(f"{src_db}.{src_schema}.CUSTOMER_PROFILES")

        customer_agg = txn.group_by("CUSTOMER_ID").agg(
            F.count("*").alias("TOTAL_TXN_COUNT"),
            F.avg("AMOUNT").alias("AVG_TXN_AMOUNT"),
            F.max("AMOUNT").alias("MAX_TXN_AMOUNT"),
            F.stddev("AMOUNT").alias("STDDEV_TXN_AMOUNT"),
            F.count_distinct("MERCHANT_ID").alias("UNIQUE_MERCHANTS"),
            F.count_distinct(F.dayofyear("TIMESTAMP")).alias("ACTIVE_DAYS"),
            F.avg(
                F.when(F.hour("TIMESTAMP") < 6, F.lit(1)).when(F.hour("TIMESTAMP") > 22, F.lit(1)).otherwise(F.lit(0))
            ).alias("LATE_NIGHT_TXN_RATIO"),
            F.max("TIMESTAMP").alias("FEATURE_TS"),
        )

        features_df = customer_agg.join(cust, on="CUSTOMER_ID", how="inner").select(
            F.col("CUSTOMER_ID"),
            F.col("TOTAL_TXN_COUNT"),
            F.col("AVG_TXN_AMOUNT"),
            F.col("MAX_TXN_AMOUNT"),
            F.col("STDDEV_TXN_AMOUNT"),
            F.col("UNIQUE_MERCHANTS"),
            F.col("ACTIVE_DAYS"),
            F.col("LATE_NIGHT_TXN_RATIO"),
            F.col("CREDIT_SCORE"),
            F.col("ACCOUNT_AGE_DAYS"),
            F.col("ANNUAL_INCOME"),
            F.col("FEATURE_TS"),
        )

        customer_fv = FeatureView(
            name="CUSTOMER_RISK_FEATURES",
            entities=[customer_entity],
            feature_df=features_df,
            timestamp_col="FEATURE_TS",
            refresh_freq="1 hour",
            desc="Customer-level risk signals for fraud detection",
        )
        fs.register_feature_view(feature_view=customer_fv, version="V1", overwrite=True)

        return json.dumps(
            {"status": "success", "step": "feature_engineering", "feature_view": "CUSTOMER_RISK_FEATURES$V1"}
        )

    return feature_engineering


def build_train_model_remote(cfg: dict):
    """Build the @remote-decorated training function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    fv = f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"
    stage = f"@{db}.{schema}.PIPELINE_STAGE"
    # Experiment tracking config (captured by closure)
    exp_enabled = EXPERIMENT_CONFIG.get("enabled", "false") == "true"
    exp_name = EXPERIMENT_CONFIG.get("experiment_name", "FRAUD_DETECTION_TRAINING")
    exp_run_prefix = EXPERIMENT_CONFIG.get("run_name_prefix", "pipeline")
    # Training hyperparameters (captured by closure from config)
    training_params = dict(TRAINING_PARAMS)

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=ML_RUNTIME_DEPS + ["snowflake-ml-python"],
    )
    def train_model() -> str:
        import json

        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
        from snowflake.ml.feature_store import CreationMode, FeatureStore
        from snowflake.snowpark import Session as _Session
        from snowflake.snowpark import functions as F

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        # Use Feature Store for point-in-time correct dataset generation
        fs = FeatureStore(
            session=session,
            database=db,
            name=schema,
            default_warehouse=wh,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )
        cust_fv = fs.get_feature_view(fv.split("$")[0], fv.split("$")[1])

        # Build spine: entity keys + timestamp + label from source
        spine = session.table(f"{src_db}.{src_schema}.RAW_TRANSACTIONS").select(
            F.col("CUSTOMER_ID"),
            F.col("TIMESTAMP"),
            F.col("IS_FRAUD"),
        )

        # Generate dataset with point-in-time correctness
        dataset = fs.generate_dataset(
            name="FRAUD_TRAINING_DATA",
            version="V1",
            spine_df=spine,
            features=[cust_fv],
            spine_timestamp_col="TIMESTAMP",
            spine_label_cols=["IS_FRAUD"],
            desc="Training dataset for fraud detection model",
        )

        df = dataset.read.to_pandas()

        feature_cols = [c for c in df.columns if c not in ("CUSTOMER_ID", "TIMESTAMP", "IS_FRAUD")]
        X = df[feature_cols].fillna(0)
        y = df["IS_FRAUD"].astype(int)
        groups = df["CUSTOMER_ID"]

        # Split by customer (group-aware) to avoid entity leakage
        # Use stratified group k-fold to get train/test indices
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        params = training_params

        # Cross-validation (group-aware)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_cv_idx, val_cv_idx in cv.split(X_train, y_train, groups.iloc[train_idx]):
            fold_model = xgb.XGBClassifier(**params)
            fold_model.fit(X_train.iloc[train_cv_idx], y_train.iloc[train_cv_idx], verbose=False)
            fold_proba = fold_model.predict_proba(X_train.iloc[val_cv_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train.iloc[val_cv_idx], fold_proba))

        # Final model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "cv_auc_mean": float(np.mean(cv_scores)),
            "feature_view": fv,
        }

        # Save model artifact to stage
        model.save_model("/tmp/model.ubj")
        session.file.put(
            "/tmp/model.ubj", f"@{db}.{schema}.PIPELINE_STAGE/artifacts/", auto_compress=False, overwrite=True
        )

        # Save sample input
        X_test.head(10).to_json("/tmp/sample_input.json", orient="records")
        session.file.put(
            "/tmp/sample_input.json", f"@{db}.{schema}.PIPELINE_STAGE/artifacts/", auto_compress=False, overwrite=True
        )

        # Experiment tracking
        if exp_enabled:
            from datetime import datetime

            from snowflake.ml.experiment import ExperimentTracking

            exp = ExperimentTracking(session=session, database_name=db, schema_name=schema)
            exp.set_experiment(exp_name)
            run_name = f"{exp_run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with exp.start_run(run_name):
                exp.log_params(params)
                exp.log_params({"feature_view": fv, "dataset_rows": str(len(df)), "test_size": "0.2"})
                # log_metrics only accepts numeric values — filter out strings
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                exp.log_metrics(numeric_metrics)

        # Write metrics to results table
        result = {"status": "success", "step": "training", "metrics": metrics}
        result_json = json.dumps(result)
        session.sql(f"""
            INSERT INTO {db}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
            SELECT 'training', 'SUCCESS', PARSE_JSON($${result_json}$$), CURRENT_TIMESTAMP()
        """).collect()

        return json.dumps(result)

    return train_model


def build_evaluate_remote(cfg: dict):
    """Build the @remote-decorated evaluation function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    stage = f"@{db}.{schema}.PIPELINE_STAGE"

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=["snowflake-ml-python"],
    )
    def evaluate_model() -> str:
        import json

        from snowflake.snowpark import Session as _Session

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        rows = session.sql(f"""
            SELECT RESULT FROM {db}.{schema}.PIPELINE_RESULTS
            WHERE STEP = 'training' AND STATUS = 'SUCCESS'
            ORDER BY CREATED_AT DESC LIMIT 1
        """).collect()

        if not rows:
            raise RuntimeError("EVALUATE failed: no training results found in PIPELINE_RESULTS")

        training_result = json.loads(rows[0]["RESULT"])
        metrics = training_result.get("metrics", {})

        result = {"status": "success", "step": "evaluation", "metrics": metrics, "pipeline_status": "READY_FOR_REVIEW"}
        result_json = json.dumps(result)
        session.sql(f"""
            INSERT INTO {db}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
            SELECT 'evaluation', 'SUCCESS', PARSE_JSON($${result_json}$$), CURRENT_TIMESTAMP()
        """).collect()

        return json.dumps(result)

    return evaluate_model


# ─── Warehouse-based alternatives (StoredProcedureCall) ──────────────────────


def feature_eng_warehouse(session: Session) -> str:
    """Feature engineering on warehouse — registers Feature Views."""
    from features.feature_views import register_feature_views

    # Use session's current context (set by the Task's database/schema/warehouse)
    db = session.get_current_database().replace('"', "")
    schema = session.get_current_schema().replace('"', "")
    wh = session.get_current_warehouse().replace('"', "")
    register_feature_views(session=session, database=db, schema=schema, warehouse=wh)
    return "feature_engineering_complete"


def train_model_warehouse(session: Session) -> str:
    """Training on warehouse is not supported — requires SPCS for custom packages."""
    raise NotImplementedError(
        "Training requires SPCS compute pool for custom packages (xgboost, sklearn). "
        "Set training_compute='spcs' in PIPELINE_CONFIG."
    )


def evaluate_warehouse(session: Session) -> str:
    """Evaluation on warehouse — reads training metrics and writes evaluation row."""
    import json

    from snowflake.snowpark.functions import col, current_timestamp, parse_json

    rows = session.sql("""
        SELECT RESULT FROM PIPELINE_RESULTS
        WHERE STEP = 'training' AND STATUS = 'SUCCESS'
        ORDER BY CREATED_AT DESC LIMIT 1
    """).collect()
    if not rows:
        raise RuntimeError("EVALUATE failed: no training results found in PIPELINE_RESULTS")

    training_result = json.loads(rows[0]["RESULT"])
    metrics = training_result.get("metrics", {})
    result = {"status": "success", "step": "evaluation", "metrics": metrics, "pipeline_status": "READY_FOR_REVIEW"}
    result_json = json.dumps(result)

    # Write evaluation row using Snowpark DataFrame API (avoids SQL escaping issues)
    from snowflake.snowpark import Row

    eval_df = session.create_dataframe(
        [Row(STEP="evaluation", STATUS="SUCCESS", RESULT=result_json)],
    )
    eval_df.select(
        col("STEP"),
        col("STATUS"),
        parse_json(col("RESULT")).alias("RESULT"),
        current_timestamp().alias("CREATED_AT"),
    ).write.mode("append").save_as_table("PIPELINE_RESULTS")
    return "evaluation_complete"


# ─── DAG Deployment ──────────────────────────────────────────────────────────


def deploy_dag(env: str):
    """Deploy the ML Training Pipeline DAG using Python SDK."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    fe_compute = PIPELINE_CONFIG.get("feature_engineering_compute", "warehouse")
    train_compute = PIPELINE_CONFIG.get("training_compute", "spcs")
    eval_compute = PIPELINE_CONFIG.get("evaluation_compute", "spcs")

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Deploying DAG: {db}.{schema}.{DAG_NAME}")
    print(f"  Feature eng: {fe_compute} | Training: {train_compute} | Evaluation: {eval_compute}")

    # Create infrastructure
    session.sql(f"CREATE STAGE IF NOT EXISTS {db}.{schema}.PIPELINE_STAGE").collect()
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.PIPELINE_RESULTS (
            STEP VARCHAR, STATUS VARCHAR, RESULT VARIANT, CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()

    # Build the DAG
    root = Root(session)
    stage_location = f"@{db}.{schema}.PIPELINE_STAGE"

    # Add schedule for periodic retraining (if enabled)
    dag_schedule = None
    if RETRAIN_CONFIG.get("enabled", "false") == "true":
        schedule_str = RETRAIN_CONFIG.get("schedule", "")
        if schedule_str:
            from snowflake.core.task import Cron

            # Parse "USING CRON <expr> <tz>" format
            parts = schedule_str.replace("USING CRON ", "").strip()
            # Split: last token is timezone, rest is cron expression
            tokens = parts.split()
            tz = tokens[-1]
            cron_expr = " ".join(tokens[:-1])
            dag_schedule = Cron(cron_expr, tz)
            print(f"  Schedule: {cron_expr} ({tz})")

    with DAG(DAG_NAME, warehouse=wh, schedule=dag_schedule) as dag:
        # Feature Engineering task
        if fe_compute == "spcs":
            fe_func = build_feature_eng_remote(cfg)
            fe_task = DAGTask("FEATURE_ENG", definition=fe_func)
        else:
            fe_task = DAGTask(
                "FEATURE_ENG",
                StoredProcedureCall(
                    feature_eng_warehouse,
                    stage_location=stage_location,
                    packages=["snowflake-ml-python", "snowflake-snowpark-python"],
                ),
                warehouse=wh,
            )

        # Training task
        if train_compute == "spcs":
            train_func = build_train_model_remote(cfg)
            train_task = DAGTask("TRAIN_MODEL", definition=train_func)
        else:
            train_task = DAGTask(
                "TRAIN_MODEL",
                StoredProcedureCall(
                    train_model_warehouse, stage_location=stage_location, packages=["snowflake-snowpark-python"]
                ),
                warehouse=wh,
            )

        # Evaluation task
        if eval_compute == "spcs":
            eval_func = build_evaluate_remote(cfg)
            eval_task = DAGTask("EVALUATE", definition=eval_func)
        else:
            eval_task = DAGTask(
                "EVALUATE",
                StoredProcedureCall(
                    evaluate_warehouse, stage_location=stage_location, packages=["snowflake-snowpark-python"]
                ),
                warehouse=wh,
            )

        # Define dependencies
        fe_task >> train_task >> eval_task

    # Deploy
    schema_ref = root.databases[db].schemas[schema]
    dag_op = DAGOperation(schema_ref)
    dag_op.deploy(dag, mode=CreateMode.or_replace)
    print(f"  DAG deployed: {DAG_NAME}")

    # Only resume root task if explicitly requested (scheduled-retrain workflow sets RESUME_SCHEDULE=true).
    # Do NOT auto-resume — otherwise cron fires during CI runs and interferes with wait_for_task.
    if os.getenv("RESUME_SCHEDULE", "false").lower() == "true" and dag_schedule:
        session.sql(f"ALTER TASK {db}.{schema}.{DAG_NAME} RESUME").collect()
        print("  Root task resumed (scheduled retraining active)")

    session.close()
    print("Deploy complete.")


def execute_dag(env: str):
    """Trigger the DAG execution."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    # Execute the root task (no longer deletes PIPELINE_RESULTS — history is preserved)
    root = Root(session)
    task_res = root.databases[db].schemas[schema].tasks[DAG_NAME]
    task_res.execute()
    print(f"Executed: {db}.{schema}.{DAG_NAME}")

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
    parser = argparse.ArgumentParser(description="ML Pipeline Task DAG (Python SDK)")
    parser.add_argument("--deploy", action="store_true", help="Deploy the Task DAG")
    parser.add_argument("--execute", action="store_true", help="Execute the Task DAG")
    parser.add_argument("--status", action="store_true", help="Show recent task history")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "dev"), choices=["dev", "stage"])
    args = parser.parse_args()

    if args.deploy:
        deploy_dag(args.env)
    elif args.execute:
        execute_dag(args.env)
    elif args.status:
        show_status(args.env)
    else:
        parser.print_help()
