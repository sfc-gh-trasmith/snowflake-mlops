"""Submit ML training job to Snowflake compute pool using @remote decorator.

This script trains the fraud detection XGBoost model on SNOW_MLOPS_DEV_POOL,
and registers the model to the Model Registry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session
from config import DATABASE, SCHEMA, WAREHOUSE, COMPUTE_POOL, JOB_STAGE, FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION

from snowflake.ml.jobs import remote

# These get captured by the @remote closure when the function is serialized
_FV_TABLE = f'"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"'
_DB = DATABASE
_SCHEMA = SCHEMA


@remote(
    COMPUTE_POOL,
    stage_name=JOB_STAGE,
    pip_requirements=[
        "xgboost",
        "scikit-learn",
        "snowflake-ml-python",
    ],
)
def train_and_register() -> str:
    """Train XGBoost fraud model and register to Model Registry.

    Reads features directly from Feature View Dynamic Tables + source data
    via SQL (avoids Dataset API serialization issues in remote containers).
    """
    import json
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

    db = _DB
    schema = _SCHEMA
    feature_table = _FV_TABLE
    model_name = "MLOPS_FRAUD_DETECTOR"

    # Auto-increment version
    try:
        versions_df = session.sql(f"SHOW VERSIONS IN MODEL {db}.{schema}.{model_name}").collect()
        existing = [r["name"] for r in versions_df]
        max_v = max(int(v.replace("V", "")) for v in existing if v.startswith("V") and v[1:].isdigit())
        version_name = f"V{max_v + 1}"
    except Exception:
        version_name = "V1"

    # Read training data by joining Feature View DT with source labels
    print(f"Loading training data from {feature_table}...")
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
        FROM {db}.{schema}.{feature_table} c
        JOIN {db}.{schema}.RAW_TRANSACTIONS t
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Hyperparameters
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
        print(f"  Fold {fold + 1}: AUC={fold_auc:.4f}")
    print(f"  CV Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

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
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    # Register model
    print(f"Registering model: {db}.{schema}.{model_name}/{version_name}")
    reg = Registry(session=session, database_name=db, schema_name=schema)
    reg.log_model(
        model=model,
        model_name=model_name,
        version_name=version_name,
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=X_test.head(10),
        comment=f"XGBoost fraud detector - features:{_FV_TABLE} | AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}",
    )
    print("  Model registered!")

    return json.dumps({"status": "success", "metrics": metrics, "version": version_name})


if __name__ == "__main__":
    print("=" * 60)
    print("DEV ML PIPELINE")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()

    # Step 1: Feature Engineering (create/update Feature Store)
    print("\n[1/2] Feature Engineering (register/update Feature Store)...")
    from features.feature_views import register_feature_views

    register_feature_views(session)

    # Step 2: Train and register model
    print("\n[2/2] Submitting training job to compute pool...")
    job = train_and_register()
    print("  Job submitted. Waiting for completion...")
    print("  (Compute pool cold-start + pip install + training takes ~5 min)")
    result = job.result()
    print(f"\n{'=' * 60}")
    print("DEV ML PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(result)
