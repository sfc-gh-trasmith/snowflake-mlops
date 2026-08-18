#!/bin/bash
# Infrastructure setup for Snowflake MLOps Framework
# Creates DEV, STAGE, and PROD environments
# Usage: SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION bash scripts/setup.sh

set -e

echo "=== Setting up Snowflake MLOps infrastructure (DEV + STAGE + PROD) ==="

snow sql -q "
-- =============================================================================
-- DEV Environment
-- =============================================================================
CREATE DATABASE IF NOT EXISTS SNOW_MLOPS_DEV;
CREATE SCHEMA IF NOT EXISTS SNOW_MLOPS_DEV.ML;

CREATE WAREHOUSE IF NOT EXISTS SNOW_MLOPS_DEV_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS SNOW_MLOPS_DEV_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_M
    AUTO_SUSPEND_SECS = 300
    AUTO_RESUME = TRUE;

USE SCHEMA SNOW_MLOPS_DEV.ML;
CREATE STAGE IF NOT EXISTS ML_ARTIFACTS ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS DAG_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS JOB_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- =============================================================================
-- STAGE Environment
-- =============================================================================
CREATE DATABASE IF NOT EXISTS SNOW_MLOPS_STAGE;
CREATE SCHEMA IF NOT EXISTS SNOW_MLOPS_STAGE.ML;

CREATE WAREHOUSE IF NOT EXISTS SNOW_MLOPS_STAGE_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS SNOW_MLOPS_STAGE_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_M
    AUTO_SUSPEND_SECS = 300
    AUTO_RESUME = TRUE;

USE SCHEMA SNOW_MLOPS_STAGE.ML;
CREATE STAGE IF NOT EXISTS ML_ARTIFACTS ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS DAG_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS JOB_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- =============================================================================
-- PROD Environment
-- =============================================================================
CREATE DATABASE IF NOT EXISTS SNOW_MLOPS_PROD;
CREATE SCHEMA IF NOT EXISTS SNOW_MLOPS_PROD.ML;

CREATE WAREHOUSE IF NOT EXISTS SNOW_MLOPS_PROD_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS SNOW_MLOPS_PROD_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_M
    AUTO_SUSPEND_SECS = 300
    AUTO_RESUME = TRUE;

USE SCHEMA SNOW_MLOPS_PROD.ML;
CREATE STAGE IF NOT EXISTS ML_ARTIFACTS ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS DAG_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS JOB_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- =============================================================================
-- Account-level grants
-- =============================================================================
GRANT EXECUTE TASK ON ACCOUNT TO ROLE ACCOUNTADMIN;
GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE ACCOUNTADMIN;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE ACCOUNTADMIN;
"

echo ""
echo "=== Infrastructure setup complete ==="
echo ""
echo "DEV:"
echo "  Database:      SNOW_MLOPS_DEV"
echo "  Warehouse:     SNOW_MLOPS_DEV_WH"
echo "  Compute Pool:  SNOW_MLOPS_DEV_POOL"
echo ""
echo "STAGE:"
echo "  Database:      SNOW_MLOPS_STAGE"
echo "  Warehouse:     SNOW_MLOPS_STAGE_WH"
echo "  Compute Pool:  SNOW_MLOPS_STAGE_POOL"
echo ""
echo "PROD:"
echo "  Database:      SNOW_MLOPS_PROD"
echo "  Warehouse:     SNOW_MLOPS_PROD_WH"
echo "  Compute Pool:  SNOW_MLOPS_PROD_POOL"
echo ""
echo "Stages (all):    ML_ARTIFACTS, DAG_STAGE, JOB_STAGE"
