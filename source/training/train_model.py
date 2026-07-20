"""Train XGBoost fraud detection model.

Designed to run as a Snowflake ML Job on a compute pool via @remote decorator.
Integrates with Experiment Tracking and returns metrics via TaskContext for DAG use.

Can also be called standalone for development/testing.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
from snowflake.ml.experiment.experiment_tracking import ExperimentTracking


DEFAULT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "scale_pos_weight": 33,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
}

FEATURE_COLUMNS = [
    # Customer features
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
    # Transaction features
    "AMOUNT",
    "AMOUNT_TO_AVG_RATIO",
    "IS_HIGH_RISK_MERCHANT",
    "MERCHANT_RISK_SCORE",
    "HOUR_OF_DAY",
    "IS_WEEKEND",
    "IS_LATE_NIGHT",
]

LABEL_COLUMN = "IS_FRAUD"


def load_training_data(session: Session, database: str, schema: str) -> pd.DataFrame:
    """Load training data from Feature Store dataset."""
    from snowflake.ml.dataset import Dataset

    dataset = Dataset.load(session=session, name=f"{database}.{schema}.FRAUD_TRAINING_DATA")
    dataset.select_version("V1")
    df = dataset.read.to_pandas()
    print(f"Loaded {len(df):,} rows, {df[LABEL_COLUMN].sum():,} fraud cases ({df[LABEL_COLUMN].mean():.2%})")
    return df


def train_model(
    df: pd.DataFrame,
    params: dict = None,
    experiment: ExperimentTracking = None,
    run_name: str = "fraud_detector_v1",
) -> tuple:
    """Train XGBoost classifier with cross-validation.

    Args:
        df: Training dataframe with features and label.
        params: XGBoost hyperparameters (uses defaults if None).
        experiment: Optional experiment tracker for logging.
        run_name: Name for the experiment run.

    Returns:
        Tuple of (trained_model, metrics_dict, feature_importance_dict)
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].fillna(0)
    y = df[LABEL_COLUMN].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if experiment:
        experiment.log_params(params)
        experiment.log_param("n_features", len(available_features))
        experiment.log_param("train_size", len(X_train))
        experiment.log_param("test_size", len(X_test))
        experiment.log_param("fraud_rate", float(y.mean()))

    # Cross-validation for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, fold_proba)
        cv_scores.append(fold_auc)

        if experiment:
            experiment.log_metric("cv_auc_roc", fold_auc, step=fold)

    print(f"CV AUC-ROC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Final model on full training set
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate on holdout test set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
    }

    # Feature importance
    importance = dict(zip(available_features, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    if experiment:
        for k, v in metrics.items():
            experiment.log_metric(k, v)

    print("\nTest Metrics:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("\nTop 5 features:")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    return model, metrics, importance


def save_model_locally(model, path: str = None) -> str:
    """Serialize model to a local pickle file."""
    if path is None:
        path = str(Path(tempfile.gettempdir()) / "fraud_detector_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {path}")
    return path


def run_training_pipeline(
    session: Session = None,
    database: str = "SNOW_MLOPS_DEV",
    schema: str = "ML",
    params: dict = None,
    run_name: str = "fraud_detector_v1",
) -> tuple:
    """Full training pipeline: load data, train, evaluate, save.

    Returns (model, metrics, model_path)
    """
    from snowpark_session import create_snowpark_session

    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE DATABASE {database}").collect()
    session.sql(f"USE SCHEMA {schema}").collect()

    # Setup experiment tracking
    experiment = ExperimentTracking(
        session=session,
        database_name=database,
        schema_name=schema,
    )
    experiment.set_experiment("FRAUD_DETECTION_EXPERIMENTS")
    experiment.start_run(run_name=run_name)

    # Load data and train
    df = load_training_data(session, database, schema)
    model, metrics, importance = train_model(df, params=params, experiment=experiment, run_name=run_name)

    # Save model
    model_path = save_model_locally(model)

    # Log model artifact
    experiment.log_model(model, model_name="fraud_detector_xgb")
    experiment.end_run()

    if close_session:
        session.close()

    return model, metrics, model_path


if __name__ == "__main__":
    run_training_pipeline()
