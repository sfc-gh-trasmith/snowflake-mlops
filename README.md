# Snowflake MLOps Framework

A reference template for building production-grade MLOps pipelines on Snowflake. This framework demonstrates how to take any ML use case from experimentation to production using Snowflake-native services -- no external ML infrastructure required.

The included demo uses a fraud detection classifier (XGBoost), but the pattern applies to any ML use case: churn prediction, demand forecasting, recommendation systems, etc. Swap the data and model code; the infrastructure, CI/CD, and deployment patterns remain the same.

## Architecture

```
feature branch → PR (lint + test) → merge to main → deploy-stage → approve → deploy-prod → tag release
```

**Environment separation is at the database level within a single Snowflake account.** Each environment (DEV, STAGE, PROD) is its own database with isolated resources (warehouses, compute pools, models, services).

| Component | Snowflake Service |
|-----------|------------------|
| Pipeline Orchestration | Snowflake Tasks (Python SDK DAG) |
| Feature Engineering | Feature Store (Dynamic Tables) |
| Model Training | ML Jobs (`@remote` on Compute Pools) |
| Model Versioning | Model Registry (auto-increment) |
| Model Promotion | Cross-database replication |
| Real-Time Serving | SPCS containers + Snowflake Gateway |
| Zero-Downtime Deploy | Blue/green with Gateway traffic shift |
| Batch Inference | Model Registry `run()` on warehouse |
| CI/CD | GitHub Actions with OIDC (zero secrets) |
| Rollback | Manual dispatch workflow |

## Pipeline: Snowflake Task DAG

The pipeline is orchestrated as a **Snowflake Task DAG** using the Python SDK (`snowflake.core.task.dagv1`). Each step runs as an **ML Job** on a compute pool (SPCS) or as a stored procedure on a warehouse -- configurable per step via `source/config.py`.

```
ML_TRAINING_PIPELINE (root)
  └── FEATURE_ENG (@remote → compute pool)
        └── TRAIN_MODEL (@remote → compute pool)
              └── EVALUATE (@remote → compute pool)
```

Configure compute mode per step in `source/config.py`:

```python
PIPELINE_CONFIG = {
    "feature_engineering_compute": "spcs",   # "warehouse" or "spcs"
    "training_compute": "spcs",
    "evaluation_compute": "spcs",
    "task_timeout_ms": "7200000",            # 2 hours max
    ...
}
```

## Quickstart

### Prerequisites

- Snowflake account with `ACCOUNTADMIN` role (for initial setup only; CI uses `MLOPS_DEPLOY_ROLE`)
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
bash scripts/setup.sh
```

This creates:
- `SNOW_MLOPS_DEV`, `SNOW_MLOPS_STAGE`, `SNOW_MLOPS_PROD` databases
- Warehouses and compute pools per environment
- Internal stages for pipeline artifacts
- `MLOPS_DEPLOY_ROLE` with appropriate grants

### 3. Generate Synthetic Data

```bash
uv run python scripts/generate_dataset.py
```

Creates 100K synthetic transactions with ~3% fraud rate in `SNOW_MLOPS_PROD.ML`.

### 4. Deploy and Run the Pipeline (DEV)

```bash
# Deploy the Task DAG to DEV
uv run python source/pipeline/ml_pipeline_dag.py --deploy --env dev

# Execute the pipeline (triggers Feature Eng → Training → Evaluation)
uv run python source/pipeline/ml_pipeline_dag.py --execute --env dev

# Check status
uv run python source/pipeline/ml_pipeline_dag.py --status --env dev
```

### 5. Set Up CI/CD

```bash
bash scripts/setup_cicd.sh
```

Configure GitHub repo variables:
- `SNOWFLAKE_ACCOUNT` - your account identifier
- `SNOWFLAKE_DATABASE_STAGE` - `SNOW_MLOPS_STAGE`
- `SNOWFLAKE_DATABASE_PROD` - `SNOW_MLOPS_PROD`
- `SNOWFLAKE_SCHEMA` - `ML`
- `SNOWFLAKE_USER_STAGE` - `SVC_GITHUB_ACTIONS_STAGE`
- `SNOWFLAKE_USER_PROD` - `SVC_GITHUB_ACTIONS`
- `TOPOLOGY` - `single-account` (default)

Configure GitHub environments:
- **STAGE** - no protection rules (auto-deploys on merge)
- **PROD** - requires reviewer approval before deploy

## Project Structure

```
snowflake-mlops/
├── .github/workflows/
│   ├── pr-checks.yml              # PR: lint + format + tests
│   ├── deploy.yml                 # Main → lint → test → deploy-stage → deploy-prod
│   └── rollback.yml               # Manual: rollback to previous model version
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
│   ├── wait_for_task.py           # Poll task DAG until completion
│   ├── quality_gate_and_register.py  # Validate metrics + register model
│   ├── run_batch_inference.py     # Batch inference validation
│   └── deploy_prod_service.py     # PROD: blue/green gateway deploy
├── source/
│   ├── config.py                  # Centralized configuration
│   ├── snowpark_session.py        # Session helper (local SSO + CI OIDC)
│   ├── features/                  # Feature Store definitions
│   │   ├── entities.py            # Entity definitions
│   │   ├── feature_views.py       # Feature View SQL + registration
│   │   └── training_data.py       # Training dataset builder
│   ├── pipeline/
│   │   └── ml_pipeline_dag.py     # Task DAG definition + ML Job functions
│   └── serving/
│       └── batch_inference.py     # Batch inference utilities
├── tests/
│   ├── test_config.py             # Unit tests (config validation)
│   └── test_endpoint.py           # Integration tests (gateway + predictions)
├── notebooks/                     # Educational Jupyter notebooks (01-05)
└── docs/
    └── docs.html                  # Detailed architecture documentation
```

## CI/CD Workflows

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `pr-checks.yml` | PR to `main` | Lint (ruff), format check, unit tests |
| `deploy.yml` | Push to `main` or manual dispatch | Full pipeline: lint → test → deploy-stage → deploy-prod |
| `rollback.yml` | Manual dispatch | Rollback to a previous model version |

### deploy.yml Pipeline

```
lint → test → deploy-stage → [approval] → deploy-prod
                  │
                  ├── Deploy Task DAG (Python SDK)
                  ├── Execute pipeline (FEATURE_ENG → TRAIN → EVALUATE)
                  ├── Wait for completion (poll TASK_HISTORY)
                  ├── Quality gate (AUC-ROC, precision, recall thresholds)
                  ├── Register model (auto-increment version)
                  └── Batch inference validation
                                          │
                                          ├── Register Feature Views in PROD
                                          ├── Batch inference (validate model.run())
                                          ├── Blue/green SPCS deploy
                                          ├── Verify gateway endpoint
                                          └── Tag release (prod/V12-20260810-201724)
```

### Release Tags

Every successful PROD deployment creates a git tag: `prod/{VERSION}-{TIMESTAMP}` (e.g., `prod/V12-20260810-201724`). This tag corresponds to the model version set as DEFAULT in `SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR`.

### Rollback

```bash
# Via GitHub Actions UI: Actions → Rollback → Run workflow
# Inputs: model_version (e.g., V11), reason (e.g., "regression in precision")
```

The rollback workflow sets the DEFAULT version, redeploys the SPCS service, and validates the endpoint.

## Key Commands

```bash
# Lint and format
uv run ruff check source/ scripts/
uv run ruff format source/ scripts/

# Run unit tests
uv run pytest tests/ -v --ignore=tests/test_endpoint.py

# Deploy Task DAG (local, DEV environment)
uv run python source/pipeline/ml_pipeline_dag.py --deploy --env dev

# Execute pipeline (local, DEV environment)
uv run python source/pipeline/ml_pipeline_dag.py --execute --env dev

# Check task history
uv run python source/pipeline/ml_pipeline_dag.py --status --env dev

# Run endpoint integration tests (requires active PROD service)
uv run pytest tests/test_endpoint.py -v
```

## Environment Strategy

| Environment | Database | Purpose | Training | Serving |
|-------------|----------|---------|----------|---------|
| DEV | `SNOW_MLOPS_DEV` | Developer experimentation | Yes (local trigger) | Optional |
| STAGE | `SNOW_MLOPS_STAGE` | Automated CI validation | Yes (CI trigger) | Never |
| PROD | `SNOW_MLOPS_PROD` | Production serving | Never | Always (Gateway) |

All environments live in a **single Snowflake account** with database-level isolation. Source data always resides in PROD; DEV and STAGE read from it for training but write artifacts to their own databases.

## Quality Gate

Models must pass all thresholds to be promoted (configured in `source/config.py`):

| Metric | Threshold |
|--------|-----------|
| AUC-ROC | >= 0.60 |
| Precision | >= 0.03 |
| Recall | >= 0.30 |

If the quality gate fails, the workflow exits and the model is not registered.

## Documentation

Open `docs/docs.html` in a browser for a detailed architecture walkthrough covering the Task DAG, ML Jobs, Feature Store, Model Registry, Gateway deployment, CI/CD, and RBAC.
