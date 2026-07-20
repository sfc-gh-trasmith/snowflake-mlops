"""Centralized configuration for the Snowflake MLOps demo."""

DATABASE = "SNOW_MLOPS_DEV"
SCHEMA = "ML"
WAREHOUSE = "SNOW_MLOPS_DEV_WH"
COMPUTE_POOL = "SNOW_MLOPS_DEV_POOL"

FULLY_QUALIFIED_SCHEMA = f"{DATABASE}.{SCHEMA}"

# Stages
ML_ARTIFACTS_STAGE = f"@{DATABASE}.{SCHEMA}.ML_ARTIFACTS"
DAG_STAGE = f"@{DATABASE}.{SCHEMA}.DAG_STAGE"
JOB_STAGE = f"@{DATABASE}.{SCHEMA}.JOB_STAGE"

# Tables
RAW_TRANSACTIONS_TABLE = f"{DATABASE}.{SCHEMA}.RAW_TRANSACTIONS"
CUSTOMER_PROFILES_TABLE = f"{DATABASE}.{SCHEMA}.CUSTOMER_PROFILES"
MERCHANT_DATA_TABLE = f"{DATABASE}.{SCHEMA}.MERCHANT_DATA"

# Model
MODEL_NAME = "FRAUD_DETECTOR"
SERVICE_NAME = "FRAUD_DETECTOR_SERVICE"

# Feature Store
FEATURE_STORE_SCHEMA = SCHEMA

# Pipeline defaults
PIPELINE_CONFIG = {
    "database": DATABASE,
    "schema": SCHEMA,
    "warehouse": WAREHOUSE,
    "compute_pool": COMPUTE_POOL,
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
}
