"""Centralized configuration for the Snowflake MLOps demo."""

# Target environment (where the pipeline WRITES: features, models, experiments, services)
DATABASE = "SNOW_MLOPS_DEV"
SCHEMA = "ML"
WAREHOUSE = "SNOW_MLOPS_DEV_WH"
COMPUTE_POOL = "SNOW_MLOPS_DEV_POOL"

# Source environment (where raw data LIVES -- always PROD)
SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"

FULLY_QUALIFIED_SCHEMA = f"{DATABASE}.{SCHEMA}"

# Stages (in target environment)
ML_ARTIFACTS_STAGE = f"@{DATABASE}.{SCHEMA}.ML_ARTIFACTS"
DAG_STAGE = f"@{DATABASE}.{SCHEMA}.DAG_STAGE"
JOB_STAGE = f"@{DATABASE}.{SCHEMA}.JOB_STAGE"

# Source tables (read-only, from PROD)
RAW_TRANSACTIONS_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS"
CUSTOMER_PROFILES_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.CUSTOMER_PROFILES"
MERCHANT_DATA_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.MERCHANT_DATA"

# Model
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"
SERVICE_NAME = "MLOPS_FRAUD_DETECTOR_SERVICE"

# Feature Store
FEATURE_STORE_SCHEMA = SCHEMA
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "V1"  # Bump when feature SQL changes

# Quality gate thresholds (model must meet ALL to promote to PROD)
MIN_AUC_ROC = 0.60
MIN_PRECISION = 0.03
MIN_RECALL = 0.30

# Pipeline defaults
PIPELINE_CONFIG = {
    "database": DATABASE,
    "schema": SCHEMA,
    "source_database": SOURCE_DATABASE,
    "source_schema": SOURCE_SCHEMA,
    "warehouse": WAREHOUSE,
    "compute_pool": COMPUTE_POOL,
    # Compute mode for each pipeline step: "warehouse" or "spcs"
    "feature_engineering_compute": "spcs",
    "training_compute": "spcs",
    "evaluation_compute": "spcs",
    # Training hyperparameters
    "n_estimators": "200",
    "learning_rate": "0.1",
    "max_depth": "6",
    "scale_pos_weight": "33",
    # Evaluation thresholds
    "min_auc_roc": "0.85",
    "min_precision": "0.70",
    "min_recall": "0.60",
    # Deployment
    "model_name": MODEL_NAME,
    "service_name": SERVICE_NAME,
    "max_instances": "2",
    # Deployment toggles
    "deploy_batch_inference": "true",
    "deploy_realtime_service": "true",
    "enable_model_monitor": "true",
    # Task configuration
    "task_timeout_ms": "7200000",  # 2 hours (max: 86400000 = 24h)
    # Internal stage for pipeline code
    "pipeline_stage": f"@{DATABASE}.{SCHEMA}.PIPELINE_STAGE",
}

# Model Monitor configuration
MONITOR_CONFIG = {
    "monitor_name": "FRAUD_DETECTOR_MONITOR",
    "function_name": "predict_proba",
    "source_table": "BATCH_PREDICTIONS",
    "timestamp_column": "PREDICTION_TS",
    "prediction_columns": ["output_feature_1"],  # P(fraud)
    "refresh_interval": "1 day",
    "aggregation_window": "7 days",
}

# Experiment Tracking configuration
EXPERIMENT_CONFIG = {
    "enabled": "true",
    "experiment_name": "FRAUD_DETECTION_TRAINING",
    "run_name_prefix": "pipeline",  # run name = prefix + timestamp
}

# Feature View refresh configuration
FEATURE_VIEW_CONFIG = {
    "customer_features_refresh": "1 hour",  # TARGET_LAG for CUSTOMER_RISK_FEATURES
    "transaction_features_refresh": "1 hour",  # TARGET_LAG for TRANSACTION_CONTEXT_FEATURES
}
