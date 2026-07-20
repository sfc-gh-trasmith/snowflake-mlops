"""Configurable ML Pipeline DAG for Fraud Detection.

Orchestrates the full ML workflow as a Snowflake Task Graph:
  PREPARE_DATA -> REFRESH_FEATURES -> TRAIN_MODEL -> EVALUATE_MODEL -> QUALITY_GATE -> [DEPLOY_MODEL | NOTIFY_ALERT]

All pipeline parameters are passed via DAG config={} and read by each task
through TaskContext.get_task_graph_config(). This enables running the same
pipeline with different configs (dev vs prod thresholds, model versions, etc.)
without code changes.

Usage:
    python ml_pipeline_dag.py --deploy     # Deploy the DAG
    python ml_pipeline_dag.py --run        # Trigger a manual run
    python ml_pipeline_dag.py --status     # Check task history
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE,
    SCHEMA,
    WAREHOUSE,
    COMPUTE_POOL,
    DAG_STAGE,
    JOB_STAGE,
    PIPELINE_CONFIG,
)
from snowpark_session import create_snowpark_session

from snowflake.core import Root
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGTaskBranch, DAGOperation
from snowflake.core.task.context import TaskContext
from snowflake.ml.jobs import remote
from snowflake.snowpark import Session


# =============================================================================
# WAREHOUSE TASKS (lightweight orchestration on warehouse)
# =============================================================================


def prepare_data(session: Session) -> str:
    """Validate source data exists and is fresh. Returns data summary."""
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()

    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]

    # Check source tables exist and have data (read from PROD)
    tables = ["RAW_TRANSACTIONS", "CUSTOMER_PROFILES", "MERCHANT_DATA"]
    summary = {}
    for table in tables:
        count = session.sql(f"SELECT COUNT(*) FROM {src_db}.{src_schema}.{table}").collect()[0][0]
        summary[table] = count
        if count == 0:
            raise ValueError(f"Table {src_db}.{src_schema}.{table} is empty!")

    ctx.set_return_value(json.dumps({"status": "ready", "table_counts": summary}))
    return json.dumps(summary)


def refresh_features(session: Session) -> str:
    """Trigger Feature Store refresh and wait for Dynamic Tables to update."""
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()
    db = cfg["database"]
    schema = cfg["schema"]

    # Force refresh of feature view Dynamic Tables
    session.sql(f"""
        ALTER DYNAMIC TABLE {db}.{schema}.CUSTOMER_RISK_FEATURES$V1 REFRESH
    """).collect()
    session.sql(f"""
        ALTER DYNAMIC TABLE {db}.{schema}.TRANSACTION_CONTEXT_FEATURES$V1 REFRESH
    """).collect()

    ctx.set_return_value(json.dumps({"status": "features_refreshed"}))
    return json.dumps({"status": "features_refreshed"})


# =============================================================================
# COMPUTE POOL TASK (heavy ML training)
# =============================================================================


@remote(
    COMPUTE_POOL,
    stage_name=JOB_STAGE,
    packages=[
        "xgboost",
        "scikit-learn",
        "snowflake-ml-python",
        "snowflake.core",
        "pandas",
        "numpy",
    ],
    database=DATABASE,
    schema=SCHEMA,
)
def train_model_remote() -> None:
    """Train XGBoost model on compute pool. Reads config from DAG."""
    session = Session.builder.getOrCreate()
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()

    db = cfg["database"]
    schema = cfg["schema"]
    model_name = cfg["model_name"]

    # Extract hyperparameters from config
    params = {
        "n_estimators": int(cfg.get("n_estimators", "200")),
        "learning_rate": float(cfg.get("learning_rate", "0.1")),
        "max_depth": int(cfg.get("max_depth", "6")),
        "scale_pos_weight": float(cfg.get("scale_pos_weight", "33")),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    # Load training data from Feature Store dataset
    from snowflake.ml.dataset import Dataset

    dataset = Dataset.load(session=session, name=f"{db}.{schema}.FRAUD_TRAINING_DATA")
    dataset.select_version("V1")
    df = dataset.read.to_pandas()

    # Feature columns
    feature_cols = [
        "TOTAL_TXN_COUNT",
        "AVG_TXN_AMOUNT",
        "MAX_TXN_AMOUNT",
        "STDDEV_TXN_AMOUNT",
        "UNIQUE_MERCHANTS",
        "HISTORICAL_FRAUD_COUNT",
        "HISTORICAL_FRAUD_RATE",
        "ACTIVE_DAYS",
        "LATE_NIGHT_TXN_RATIO",
        "CREDIT_SCORE",
        "ACCOUNT_AGE_DAYS",
        "ANNUAL_INCOME",
        "AMOUNT",
        "AMOUNT_TO_AVG_RATIO",
        "IS_HIGH_RISK_MERCHANT",
        "MERCHANT_RISK_SCORE",
        "HOUR_OF_DAY",
        "IS_WEEKEND",
        "IS_LATE_NIGHT",
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].fillna(0)
    y = df["IS_FRAUD"].astype(int)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train
    import xgboost as xgb

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Register model
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=db, schema_name=schema)
    version_name = cfg.get("model_version", "V1")

    reg.log_model(
        model=model,
        model_name=model_name,
        version_name=version_name,
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=X_test.head(10),
        comment=f"Pipeline trained: AUC={metrics['auc_roc']:.4f}",
    )

    ctx.set_return_value(
        json.dumps(
            {
                "metrics": metrics,
                "model_name": model_name,
                "version": version_name,
                "n_features": len(available_features),
                "train_size": len(X_train),
            }
        )
    )


# =============================================================================
# EVALUATION + QUALITY GATE (warehouse tasks)
# =============================================================================


def evaluate_model_task(session: Session) -> str:
    """Read metrics from training and format evaluation report."""
    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    ctx.set_return_value(json.dumps(train_result))
    return json.dumps(train_result)


def quality_gate(session: Session) -> str:
    """Check model metrics against configurable thresholds.

    Returns task name for next step:
      - "DEPLOY_MODEL" if all thresholds pass
      - "NOTIFY_ALERT" if any threshold fails
    """
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()
    eval_result = json.loads(ctx.get_predecessor_return_value("EVALUATE_MODEL"))
    metrics = eval_result["metrics"]

    min_auc = float(cfg.get("min_auc_roc", "0.85"))
    min_precision = float(cfg.get("min_precision", "0.70"))
    min_recall = float(cfg.get("min_recall", "0.60"))

    passed = metrics["auc_roc"] >= min_auc and metrics["precision"] >= min_precision and metrics["recall"] >= min_recall

    if passed:
        return "DEPLOY_MODEL"
    return "NOTIFY_ALERT"


def deploy_model_task(session: Session) -> str:
    """Deploy model to inference service if quality gate passes."""
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()
    eval_result = json.loads(ctx.get_predecessor_return_value("EVALUATE_MODEL"))

    db = cfg["database"]
    schema = cfg["schema"]
    model_name = cfg["model_name"]
    service_name = cfg["service_name"]
    compute_pool = cfg.get("compute_pool", COMPUTE_POOL)
    max_instances = int(cfg.get("max_instances", "2"))
    version = eval_result.get("version", "V1")

    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=db, schema_name=schema)
    model = reg.get_model(model_name)
    mv = model.version(version)

    # Set as default version
    model.default = version

    # Deploy or update service
    try:
        mv.create_service(
            service_name=service_name,
            service_compute_pool=compute_pool,
            image_build_compute_pool=compute_pool,
            ingress_enabled=True,
            max_instances=max_instances,
            gpu_requests=None,
        )
        status = "created"
    except Exception as e:
        if "already exists" in str(e).lower():
            status = "already_exists_skipped"
        else:
            raise

    result = {
        "status": status,
        "service": f"{db}.{schema}.{service_name}",
        "version": version,
        "metrics": eval_result["metrics"],
    }
    ctx.set_return_value(json.dumps(result))
    return json.dumps(result)


def notify_alert(session: Session) -> str:
    """Send alert when model fails quality gate."""
    ctx = TaskContext(session)
    cfg = ctx.get_task_graph_config()
    eval_result = json.loads(ctx.get_predecessor_return_value("EVALUATE_MODEL"))

    alert_msg = (
        f"FRAUD_DETECTION_PIPELINE: Model failed quality gate.\n"
        f"Metrics: {json.dumps(eval_result.get('metrics', {}), indent=2)}\n"
        f"Thresholds: AUC>={cfg.get('min_auc_roc')}, "
        f"Precision>={cfg.get('min_precision')}, Recall>={cfg.get('min_recall')}"
    )

    # Log to a table for visibility
    session.sql(f"""
        INSERT INTO {cfg["database"]}.{cfg["schema"]}.PIPELINE_ALERTS (ALERT_TIME, MESSAGE)
        SELECT CURRENT_TIMESTAMP(), '{alert_msg.replace("'", "''")}'
    """).collect()

    ctx.set_return_value(json.dumps({"alert": "sent", "message": alert_msg}))
    return json.dumps({"alert": "sent"})


def cleanup_task(session: Session) -> None:
    """Finalizer: runs regardless of success/failure. Clean up temp artifacts."""
    pass


# =============================================================================
# DAG DEFINITION
# =============================================================================


def build_dag(config: dict = None) -> DAG:
    """Build the ML pipeline DAG with the given configuration."""
    pipeline_config = {**PIPELINE_CONFIG, **(config or {})}

    dag = DAG(
        name="FRAUD_DETECTION_PIPELINE",
        schedule=None,  # Manual for dev; use Cron("0 6 * * *", "UTC") for prod
        stage_location=DAG_STAGE,
        use_func_return_value=True,
        config=pipeline_config,
    )

    with dag:
        prepare = DAGTask("PREPARE_DATA", prepare_data, warehouse=WAREHOUSE)
        refresh = DAGTask("REFRESH_FEATURES", refresh_features, warehouse=WAREHOUSE)
        train = DAGTask("TRAIN_MODEL", definition=train_model_remote)
        evaluate = DAGTask("EVALUATE_MODEL", evaluate_model_task, warehouse=WAREHOUSE)
        gate = DAGTaskBranch("QUALITY_GATE", quality_gate, warehouse=WAREHOUSE)
        deploy = DAGTask("DEPLOY_MODEL", deploy_model_task, warehouse=WAREHOUSE)
        alert = DAGTask("NOTIFY_ALERT", notify_alert, warehouse=WAREHOUSE)
        DAGTask("CLEANUP", cleanup_task, warehouse=WAREHOUSE, is_finalizer=True)

        prepare >> refresh >> train >> evaluate >> gate >> [deploy, alert]

    return dag


def deploy_dag(session=None, config: dict = None):
    """Deploy the pipeline DAG to Snowflake."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    # Create alerts table if needed
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.PIPELINE_ALERTS (
            ALERT_TIME TIMESTAMP_NTZ,
            MESSAGE VARCHAR
        )
    """).collect()

    dag = build_dag(config)

    root = Root(session)
    dag_op = DAGOperation(root.databases[DATABASE].schemas[SCHEMA])
    dag_op.deploy(dag, mode="orReplace")

    print(f"DAG deployed: {DATABASE}.{SCHEMA}.FRAUD_DETECTION_PIPELINE")
    print(
        "  Tasks: PREPARE_DATA -> REFRESH_FEATURES -> TRAIN_MODEL -> EVALUATE_MODEL -> QUALITY_GATE -> [DEPLOY_MODEL | NOTIFY_ALERT]"
    )
    print("  Schedule: Manual (trigger with dag_op.run(dag))")
    print("\nConfig:")
    for k, v in (config or PIPELINE_CONFIG).items():
        print(f"  {k}: {v}")

    if close_session:
        session.close()


def run_dag(session=None, config: dict = None):
    """Trigger a manual run of the pipeline."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    dag = build_dag(config)
    root = Root(session)
    dag_op = DAGOperation(root.databases[DATABASE].schemas[SCHEMA])
    dag_op.run(dag)

    print("Pipeline triggered! Monitor progress:")
    print(
        f"  SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) WHERE DATABASE_NAME = '{DATABASE}' ORDER BY SCHEDULED_TIME DESC LIMIT 20;"
    )

    if close_session:
        session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline DAG")
    parser.add_argument("--deploy", action="store_true", help="Deploy the DAG")
    parser.add_argument("--run", action="store_true", help="Trigger a manual run")
    parser.add_argument("--status", action="store_true", help="Check task history")
    args = parser.parse_args()

    if args.deploy:
        deploy_dag()
    elif args.run:
        run_dag()
    elif args.status:
        session = create_snowpark_session()
        session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
        result = session.sql(f"""
            SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME, ERROR_MESSAGE
            FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY())
            WHERE DATABASE_NAME = '{DATABASE}'
            ORDER BY SCHEDULED_TIME DESC LIMIT 20
        """).show()
        session.close()
    else:
        parser.print_help()
