# Snowflake MLOps Framework

A reference template for building production-grade MLOps pipelines on Snowflake. This framework demonstrates how to take any ML use case from experimentation to production using Snowflake-native services -- no external ML infrastructure required.

The included demo uses a fraud detection classifier (XGBoost), but the pattern applies to any ML use case: churn prediction, demand forecasting, recommendation systems, etc. Swap the data and model code; the infrastructure, CI/CD, and deployment patterns remain the same.

## Architecture

```
DEV (experiment) → STAGE (automated CI) → PROD (serving via Gateway)
```

**Environment separation is at the database level within a single Snowflake account.** Each environment (DEV, STAGE, PROD) is its own database with isolated resources (warehouses, compute pools, models, services). This keeps things simple while providing clear RBAC boundaries.

| Component | Snowflake Service |
|-----------|------------------|
| Feature Engineering | Dynamic Tables (Feature Store) |
| Model Training | ML Jobs (`@remote` on Compute Pools) |
| Model Versioning | Model Registry (auto-increment) |
| Model Promotion | Cross-database replication |
| Real-Time Serving | SPCS containers + Snowflake Gateway |
| Zero-Downtime Deploy | Blue/green with Gateway traffic shift |
| CI/CD | GitHub Actions with OIDC (zero secrets) |

## Quickstart

### Prerequisites

- Snowflake account with `ACCOUNTADMIN` role
- Python 3.12+ with [uv](https://docs.astral.sh/uv/) installed
- GitHub CLI (`gh`) installed
- Snowflake CLI (`snow`) installed and configured

### 1. Clone and Install

```bash
git clone https://github.com/sfc-gh-trasmith/snowflake-mlops.git
cd snowflake-mlops
uv sync
```

### 2. Create Snowflake Infrastructure

```bash
# Creates databases, schemas, warehouses, compute pools, and stages
# for DEV, STAGE, and PROD environments
bash scripts/setup.sh
```

This creates:
- `SNOW_MLOPS_DEV`, `SNOW_MLOPS_STAGE`, `SNOW_MLOPS_PROD` databases
- Warehouses and compute pools per environment
- Internal stages for job artifacts

### 3. Generate Synthetic Data

```bash
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python scripts/generate_dataset.py
```

Creates 100K synthetic transactions with ~3% fraud rate in `SNOW_MLOPS_PROD.ML`.

### 4. Set Up Feature Store

```bash
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python -c "
import sys; sys.path.insert(0, 'source')
from snowpark_session import create_snowpark_session
from features.feature_views import create_feature_views
session = create_snowpark_session()
create_feature_views(session)
session.close()
"
```

### 5. Train a Model (DEV)

```bash
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python scripts/run_training_job.py
```

Submits training to `SNOW_MLOPS_DEV_POOL`. Takes ~5 minutes (compute pool cold-start + training). Registers model to `SNOW_MLOPS_DEV.ML.MLOPS_FRAUD_DETECTOR`.

### 6. Set Up CI/CD (Optional)

```bash
# Creates OIDC service users, network policy, and branch protection
bash scripts/setup_cicd.sh
```

Configure GitHub repo variables:
- `SNOWFLAKE_ACCOUNT` - your account identifier
- `SNOWFLAKE_DATABASE_STAGE` - `SNOW_MLOPS_STAGE`
- `SNOWFLAKE_DATABASE_PROD` - `SNOW_MLOPS_PROD`
- `SNOWFLAKE_SCHEMA` - `ML`
- `TOPOLOGY` - `single-account` (default; also supports `multi-account`, `cross-region`)

## Project Structure

```
snowflake-mlops/
├── .github/workflows/
│   ├── pr-checks.yml              # PR: lint + format + tests
│   ├── deploy.yml                 # STAGE: train + register + replicate (auto on merge)
│   └── deploy-prod.yml            # PROD: blue/green deploy (manual + approval)
├── deploy/                        # Promotion strategies (topology-aware)
│   ├── promote.py                 # CLI dispatcher (reads TOPOLOGY env var)
│   └── strategies/
│       ├── single_account.py      # Default: cross-DB replication in one account
│       ├── multi_account.py       # Future: cross-account promotion
│       └── cross_region.py        # Future: cross-region replication
├── scripts/
│   ├── setup.sh                   # Create Snowflake infrastructure
│   ├── setup_cicd.sh              # OIDC users + network policy
│   ├── generate_dataset.py        # Synthetic fraud data
│   ├── run_training_job.py        # DEV: train on compute pool
│   ├── run_stage_pipeline.py      # STAGE: train + promote (uses TOPOLOGY)
│   └── deploy_prod_service.py     # PROD: blue/green gateway deploy
├── source/
│   ├── config.py                  # Centralized configuration
│   ├── snowpark_session.py        # Session helper (local + OIDC)
│   ├── features/                  # Feature Store definitions
│   ├── training/                  # Training utilities
│   └── pipeline/                  # Snowflake Task DAG (deployed by CI)
├── tests/
│   ├── test_config.py             # Unit tests
│   └── test_endpoint.py           # Integration tests (gateway + predictions)
└── docs/
    └── mlops-architecture.html    # Detailed architecture documentation
```

## CI/CD Workflows

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `pr-checks.yml` | PR to `main` | Lint, format check, unit tests |
| `deploy.yml` | Push to `main` | Train on STAGE pool, register model, replicate to PROD |
| `deploy-prod.yml` | Manual dispatch | Blue/green: create service, health check, shift gateway |

## Key Commands

```bash
# Run linting
uv run ruff check source/ scripts/

# Run tests
uv run pytest tests/ -v

# Run DEV training pipeline
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python scripts/run_training_job.py

# Run STAGE pipeline (normally done by CI)
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python scripts/run_stage_pipeline.py

# Deploy to PROD (normally done by CI)
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run python scripts/deploy_prod_service.py

# Run endpoint integration tests
SNOWFLAKE_CONNECTION_NAME=$YOUR_CONNECTION uv run pytest tests/test_endpoint.py -v
```

## Environment Strategy

| Environment | Database | Purpose | Training | Serving |
|-------------|----------|---------|----------|--------|
| DEV | `SNOW_MLOPS_DEV` | Developer experimentation | Yes | Optional |
| STAGE | `SNOW_MLOPS_STAGE` | Automated CI validation | Yes | Never |
| PROD | `SNOW_MLOPS_PROD` | Production serving | Never | Always (Gateway) |

All environments live in a **single Snowflake account** with database-level isolation. Source data always resides in PROD; DEV and STAGE read from it for training but write artifacts to their own databases.

## Future Work

- Multi-account MLOps (separate accounts per environment)
- Cross-region model replication
- A/B testing via Gateway traffic splitting (canary deployments)
- Online Feature Store with Snowflake Postgres
- Model monitoring and drift detection

## Documentation

Open `docs/mlops-architecture.html` in a browser for a detailed Level 300 walkthrough of the entire MLOps workflow, including Feature Store, ML Jobs, Model Registry, Gateway deployment, and CI/CD orchestration.
