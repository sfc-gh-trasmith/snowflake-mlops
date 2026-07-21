"""STAGE ML Pipeline: Train, Register, and Replicate Model to PROD.

This script is executed by the STAGE GitHub Actions workflow.
Steps:
  1. Train XGBoost fraud model on SNOW_MLOPS_STAGE_POOL (@remote)
  2. Register model to SNOW_MLOPS_STAGE.ML
  3. Copy (replicate) model to SNOW_MLOPS_PROD.ML

Source data is always read from SNOW_MLOPS_PROD.ML (production tables).
No inference service is deployed in STAGE -- that only lives in PROD.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

from snowflake.ml.jobs import remote

# --- STAGE Environment Config ---
STAGE_DATABASE = "SNOW_MLOPS_STAGE"
STAGE_SCHEMA = "ML"
STAGE_WAREHOUSE = "SNOW_MLOPS_STAGE_WH"
STAGE_COMPUTE_POOL = "SNOW_MLOPS_STAGE_POOL"
STAGE_JOB_STAGE = f"@{STAGE_DATABASE}.{STAGE_SCHEMA}.JOB_STAGE"

# Feature Store version (must match config.py)
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "V1"
_FV_TABLE = f'"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"'

# Where data lives (always PROD)
SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"

# Where model gets replicated to
PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"

MODEL_NAME = "MLOPS_FRAUD_DETECTOR"


@remote(
    STAGE_COMPUTE_POOL,
    stage_name=STAGE_JOB_STAGE,
    pip_requirements=[
        "xgboost",
        "scikit-learn",
        "snowflake-ml-python",
    ],
)
def train_and_register_stage() -> str:
    """Train XGBoost fraud model on STAGE and register to STAGE Model Registry."""
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
    )
    from snowflake.snowpark import Session
    from snowflake.ml.registry import Registry

    session = Session.builder.getOrCreate()

    import subprocess

    db = "SNOW_MLOPS_STAGE"
    schema = "ML"
    source_db = "SNOW_MLOPS_PROD"
    source_schema = "ML"
    model_name = "MLOPS_FRAUD_DETECTOR"

    # Auto-increment version: query existing versions and compute next
    try:
        versions_df = session.sql(f"SHOW VERSIONS IN MODEL {db}.{schema}.{model_name}").collect()
        existing = [r["name"] for r in versions_df]
        max_v = max(int(v.replace("V", "")) for v in existing if v.startswith("V") and v[1:].isdigit())
        version_name = f"V{max_v + 1}"
    except Exception:
        version_name = "V1"

    # Get git SHA for traceability (may not be available in container)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    # Read training data from PROD source + STAGE feature views
    print(f"Loading training data from {_FV_TABLE}...")
    df = session.sql(f"""
        SELECT
            c.CUSTOMER_ID,
            c.TOTAL_TXN_COUNT,
            c.AVG_TXN_AMOUNT,
            c.MAX_TXN_AMOUNT,
            c.STDDEV_TXN_AMOUNT,
            c.UNIQUE_MERCHANTS,
            c.HISTORICAL_FRAUD_COUNT,
            c.HISTORICAL_FRAUD_RATE,
            c.ACTIVE_DAYS,
            c.LATE_NIGHT_TXN_RATIO,
            c.CREDIT_SCORE,
            c.ACCOUNT_AGE_DAYS,
            c.ANNUAL_INCOME,
            t.IS_FRAUD
        FROM {db}.{schema}.{_FV_TABLE} c
        JOIN {source_db}.{source_schema}.RAW_TRANSACTIONS t
            ON c.CUSTOMER_ID = t.CUSTOMER_ID
    """).to_pandas()
    print(f"  Loaded {len(df):,} rows, fraud rate: {df['IS_FRAUD'].mean():.2%}")

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
    ]
    X = df[feature_cols].fillna(0)
    y = df["IS_FRAUD"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "scale_pos_weight": 33,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
        fold_proba = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_proba)
        cv_scores.append(fold_auc)
    print(f"  CV Mean AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Final model
    print("Training final model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "cv_auc_mean": float(np.mean(cv_scores)),
    }
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")

    # Register model to STAGE
    print(f"Registering model: {db}.{schema}.{model_name}/{version_name}")
    reg = Registry(session=session, database_name=db, schema_name=schema)
    reg.log_model(
        model=model,
        model_name=model_name,
        version_name=version_name,
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=X_test.head(10),
        comment=f"STAGE pipeline: features:{_FV_TABLE} | AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f} | git:{git_sha}",
    )
    print("  Model registered in STAGE!")

    return json.dumps({"status": "success", "metrics": metrics, "version": version_name})


def replicate_model_to_prod(session, version: str):
    """Copy the registered model version from STAGE to PROD using the active promotion strategy."""
    import importlib
    import os

    topology = os.getenv("TOPOLOGY", "single-account")
    strategy_map = {
        "single-account": "deploy.strategies.single_account",
        "multi-account": "deploy.strategies.multi_account",
        "cross-region": "deploy.strategies.cross_region",
    }

    module_path = strategy_map.get(topology, strategy_map["single-account"])
    print(f"\nPromotion strategy: {topology}")

    # Add project root to path for deploy module import
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module = importlib.import_module(module_path)
    module.promote(version=version, session=session)


def main():
    print("=" * 60)
    print("STAGE ML PIPELINE")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {STAGE_WAREHOUSE}").collect()
    session.sql(f"USE DATABASE {STAGE_DATABASE}").collect()
    session.sql(f"USE SCHEMA {STAGE_SCHEMA}").collect()

    # Step 1: Train and register model on STAGE compute pool
    print("\n[1/2] Training model on STAGE compute pool...")
    job = train_and_register_stage()
    print("  Job submitted. Waiting for completion...")
    result = job.result()
    print(f"  Training result: {result}")

    # Extract version from result
    result_data = json.loads(result)
    version = result_data.get("version", "V1")

    # Step 2: Replicate model from STAGE to PROD
    print(f"\n[2/2] Replicating model {version} to PROD...")
    replicate_model_to_prod(session, version)

    print("\n" + "=" * 60)
    print("STAGE PIPELINE COMPLETE")
    print("  Model trained and registered in SNOW_MLOPS_STAGE.ML")
    print("  Model replicated to SNOW_MLOPS_PROD.ML")
    print("  Ready for PROD service deployment (manual dispatch)")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
