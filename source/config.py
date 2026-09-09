"""Centralized configuration for the Snowflake MLOps template."""

# Target environment (where the pipeline WRITES: features, models, experiments, services)
DATABASE = "SNOW_MLOPS_DEV"
SCHEMA = "ML"
WAREHOUSE = "SNOW_MLOPS_DEV_WH"
COMPUTE_POOL = "SNOW_MLOPS_DEV_POOL"

# Source environment (where raw data LIVES -- always PROD)
SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"

# Model
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"

# Feature Store
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "V1"  # Bump when feature SQL changes

# Quality gate thresholds (model must meet ALL to promote to PROD)
MIN_AUC_ROC = 0.01
MIN_PRECISION = 0.01
MIN_RECALL = 0.01

# Training hyperparameters (read by the training closure)
TRAINING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "scale_pos_weight": 33,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
}

# ML runtime dependencies — read from pyproject.toml [dependency-groups] ml-runtime
# Single source of truth for train/serve version alignment.
import tomllib
from pathlib import Path


def _read_ml_runtime_deps() -> list[str]:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["dependency-groups"]["ml-runtime"]


ML_RUNTIME_DEPS = _read_ml_runtime_deps()

# Pipeline configuration
PIPELINE_CONFIG = {
    "database": DATABASE,
    "schema": SCHEMA,
    "source_database": SOURCE_DATABASE,
    "source_schema": SOURCE_SCHEMA,
    "warehouse": WAREHOUSE,
    "compute_pool": COMPUTE_POOL,
    "model_name": MODEL_NAME,
    # Compute mode for each pipeline step: "warehouse" or "spcs"
    "feature_engineering_compute": "spcs",
    "training_compute": "spcs",
    "evaluation_compute": "warehouse",
}

# Model Monitor configuration
MONITOR_CONFIG = {
    "monitor_name": "FRAUD_DETECTOR_MONITOR",
    "function_name": "predict_proba",
    "source_table": "BATCH_PREDICTIONS",
    "timestamp_column": "PREDICTION_TS",
    "prediction_columns": ["output_feature_1"],
    "refresh_interval": "1 day",
    "aggregation_window": "7 days",
}

# Experiment Tracking configuration
EXPERIMENT_CONFIG = {
    "enabled": "true",
    "experiment_name": "FRAUD_DETECTION_TRAINING",
    "run_name_prefix": "pipeline",
}

# Feature View refresh configuration
FEATURE_VIEW_CONFIG = {
    "customer_features_refresh": "1 hour",
    "transaction_features_refresh": "1 hour",
}

# Scheduled Retraining configuration
RETRAIN_CONFIG = {
    "enabled": "true",
    "schedule": "USING CRON 0 6 * * MON America/Los_Angeles",
    "stage_only": "true",
    "notify_github_issue": "true",
}
