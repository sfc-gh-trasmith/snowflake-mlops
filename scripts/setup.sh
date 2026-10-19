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


-- =============================================================================
-- PyPI egress for ML Jobs
-- =============================================================================
-- ML Jobs pip-install the exact versions pinned in pyproject.toml [ml-runtime].
-- Without this integration pip cannot reach PyPI and the job silently keeps
-- the runtime image's preinstalled versions (the pipeline's version check
-- then fails the step). Name must match pypi_access_integration in config.yml.
CREATE NETWORK RULE IF NOT EXISTS SNOW_MLOPS_PROD.ML.MLOPS_PYPI_NETWORK_RULE
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('pypi.org', 'pypi.python.org', 'pythonhosted.org', 'files.pythonhosted.org');

CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS MLOPS_PYPI_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (SNOW_MLOPS_PROD.ML.MLOPS_PYPI_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'PyPI egress for snowflake-mlops ML Jobs (pinned pip_requirements)';

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
echo "PyPI access:     MLOPS_PYPI_ACCESS_INTEGRATION (ML Jobs install pinned versions)"
