"""Model Training task — trains XGBoost fraud detector and saves artifact + metrics.

Runs as a Snowflake ML Job on SPCS compute pool (or warehouse if configured).
Called by the ML_PIPELINE_TRAIN task in the Task DAG.

Does NOT register the model — that happens in GH Actions after quality gate approval.
Saves: model artifact to stage, metrics to PIPELINE_RESULTS table.
"""

import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from snowflake.snowpark import Session


def main():
    session = Session.builder.getOrCreate()

    database = os.getenv("PIPELINE_DATABASE", "SNOW_MLOPS_STAGE")
    schema = os.getenv("PIPELINE_SCHEMA", "ML")
    warehouse = os.getenv("PIPELINE_WAREHOUSE", f"{database}_WH")
    source_database = os.getenv("PIPELINE_SOURCE_DATABASE", "SNOW_MLOPS_PROD")
    source_schema = os.getenv("PIPELINE_SOURCE_SCHEMA", "ML")
    feature_view = os.getenv("PIPELINE_FEATURE_VIEW", "CUSTOMER_RISK_FEATURES$V1")

    session.sql(f"USE WAREHOUSE {warehouse}").collect()

    fv_table = f'"{feature_view}"'
    print(f"Training: loading data from {database}.{schema}.{fv_table}")

    df = session.sql(f"""
        SELECT
            c.CUSTOMER_ID,
            c.TOTAL_TXN_COUNT, c.AVG_TXN_AMOUNT, c.MAX_TXN_AMOUNT,
            c.STDDEV_TXN_AMOUNT, c.UNIQUE_MERCHANTS, c.HISTORICAL_FRAUD_COUNT,
            c.HISTORICAL_FRAUD_RATE, c.ACTIVE_DAYS, c.LATE_NIGHT_TXN_RATIO,
            c.CREDIT_SCORE, c.ACCOUNT_AGE_DAYS, c.ANNUAL_INCOME,
            t.IS_FRAUD
        FROM {database}.{schema}.{fv_table} c
        JOIN {source_database}.{source_schema}.RAW_TRANSACTIONS t
            ON c.CUSTOMER_ID = t.CUSTOMER_ID
    """).to_pandas()
    print(f"  Loaded {len(df):,} rows, fraud rate: {df['IS_FRAUD'].mean():.2%}")

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
        "n_estimators": int(os.getenv("PIPELINE_N_ESTIMATORS", "200")),
        "learning_rate": float(os.getenv("PIPELINE_LEARNING_RATE", "0.1")),
        "max_depth": int(os.getenv("PIPELINE_MAX_DEPTH", "6")),
        "scale_pos_weight": int(os.getenv("PIPELINE_SCALE_POS_WEIGHT", "33")),
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
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_view": feature_view,
    }
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")

    # Save model artifact to stage for later registration
    model_path = "/tmp/model.ubj"
    model.save_model(model_path)
    stage_path = f"@{database}.{schema}.PIPELINE_STAGE/artifacts/"
    session.file.put(model_path, stage_path, auto_compress=False, overwrite=True)
    print(f"  Model artifact saved to {stage_path}model.ubj")

    # Save sample input for model registration
    sample_path = "/tmp/sample_input.json"
    X_test.head(10).to_json(sample_path, orient="records")
    session.file.put(sample_path, stage_path, auto_compress=False, overwrite=True)

    # Write metrics to results table
    result = {"status": "success", "step": "training", "metrics": metrics}
    result_json = json.dumps(result).replace("'", "''")
    session.sql(f"""
        INSERT INTO {database}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
        VALUES ('training', 'SUCCESS', PARSE_JSON('{result_json}'), CURRENT_TIMESTAMP())
    """).collect()

    print("Training complete. Model artifact + metrics saved.")
    session.close()


if __name__ == "__main__":
    main()
