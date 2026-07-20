# MLOps Workflow Architecture

## Overview

This project implements an end-to-end MLOps pipeline for fraud detection on Snowflake, covering model development, multi-environment promotion, and production serving. The system uses three isolated Snowflake databases (DEV, STAGE, PROD) with CI/CD automation via GitHub Actions.

**Use case:** Real-time fraud detection for financial transactions. Given a customer's profile and transaction history, predict the probability of a transaction being fraudulent.

**Model:** XGBoost binary classifier trained on customer-level aggregated features from a Feature Store.

---

## Architecture Diagram

```
                    Source Data (PROD)
                    SNOW_MLOPS_PROD.ML
                 ┌────────────────────────┐
                 │  RAW_TRANSACTIONS      │
                 │  CUSTOMER_PROFILES     │
                 │  MERCHANT_DATA         │
                 └──────────┬─────────────┘
                            │ (read-only)
              ┌─────────────┼─────────────────────┐
              │             │                     │
              ▼             ▼                     ▼
    ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │      DEV        │  │      STAGE       │  │      PROD        │
    │ SNOW_MLOPS_DEV  │  │ SNOW_MLOPS_STAGE │  │ SNOW_MLOPS_PROD  │
    ├─────────────────┤  ├──────────────────┤  ├──────────────────┤
    │ Feature Store   │  │ Feature Store    │  │ Model (replica)  │
    │ Training Data   │  │ Training         │  │ Inference Service│
    │ Model Registry  │  │ Model Registry   │  │ REST Gateway     │
    │ Inference Svc   │  │ (no service)     │  │                  │
    │ Experiments     │  │                  │  │                  │
    └─────────────────┘  └───────┬──────────┘  └──────────────────┘
                                 │                       ▲
                                 │  CREATE MODEL...      │
                                 │  FROM MODEL           │
                                 └───────────────────────┘
                                   (model replication)
```

---

## Environment Separation

| Environment | Database | Purpose | Writes | Reads From |
|-------------|----------|---------|--------|------------|
| **DEV** | `SNOW_MLOPS_DEV` | Local development and experimentation | Feature Store, models, experiments, services | PROD source tables |
| **STAGE** | `SNOW_MLOPS_STAGE` | CI validation (automated by GitHub Actions) | Feature Store, models | PROD source tables |
| **PROD** | `SNOW_MLOPS_PROD` | Production serving | Inference service only | Own replicated model |

**Key design principle:** Source data always lives in PROD. DEV and STAGE read from PROD for training, but write artifacts (features, models, experiments) to their own databases. PROD never trains -- it only receives model replicas from STAGE and deploys inference services.

---

## ML Pipeline Steps

### 1. Feature Store (Dynamic Tables)

Customer-level risk features are computed as Snowflake Dynamic Tables that auto-refresh hourly:

| Feature | Description |
|---------|-------------|
| `TOTAL_TXN_COUNT` | Total transactions for customer |
| `AVG_TXN_AMOUNT` | Average transaction amount |
| `MAX_TXN_AMOUNT` | Maximum single transaction |
| `STDDEV_TXN_AMOUNT` | Spending volatility |
| `UNIQUE_MERCHANTS` | Merchant diversity |
| `HISTORICAL_FRAUD_COUNT` | Prior fraud incidents |
| `HISTORICAL_FRAUD_RATE` | Historical fraud percentage |
| `ACTIVE_DAYS` | Days with at least one transaction |
| `LATE_NIGHT_TXN_RATIO` | Proportion of late-night transactions |
| `CREDIT_SCORE` | Customer credit score |
| `ACCOUNT_AGE_DAYS` | Account tenure |
| `ANNUAL_INCOME` | Reported annual income |

### 2. Model Training (ML Job on Compute Pool)

Training runs on Snowflake compute pools using the `@remote` decorator:

- **Algorithm:** XGBoost with `scale_pos_weight=33` for class imbalance (3% fraud rate)
- **Validation:** 5-fold stratified cross-validation
- **Metrics:** AUC-ROC, PR-AUC, Precision, Recall, F1
- **Compute:** `CPU_X64_M` instance family, auto-scaling 1-3 nodes

### 3. Model Registry

Trained models are registered with:
- Version tracking (V1, V2, ...)
- Dependency declaration (`conda_dependencies=["xgboost", "scikit-learn"]`)
- Input schema inference from sample data
- Deployment metadata (comment with metrics)

### 4. Model Replication (STAGE to PROD)

```sql
CREATE MODEL SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR
  WITH VERSION V1
  FROM MODEL SNOW_MLOPS_STAGE.ML.MLOPS_FRAUD_DETECTOR
  VERSION V1;
```

This is an intra-account, cross-database copy. No replication groups needed. The model object, dependencies, and serving configuration are all copied.

### 5. Inference Service (SPCS with REST Gateway)

The model is deployed as a containerized inference service on Snowpark Container Services:

- **Ingress:** Public REST endpoint (`ingress_enabled=True`)
- **Scaling:** Up to 2 instances
- **Compute:** CPU-based (XGBoost doesn't need GPU)
- **Auto-suspend:** 1800 seconds (30 min)
- **Methods:** `predict` (binary 0/1) and `predict_proba` (probability scores)

---

## GitHub Actions CI/CD Workflows

### Workflow Files

| File | Trigger | Environment | What it Does |
|------|---------|-------------|-------------|
| `pr-checks.yml` | PR to `main` | None | Lint (ruff), format check, unit tests (pytest) |
| `deploy.yml` | Push to `main` | STAGE | Full ML pipeline: train, register, replicate to PROD |
| `deploy-prod.yml` | Manual dispatch | PROD | Deploy inference service from replicated model |

### Authentication

All workflows use **OIDC workload identity** (zero stored secrets):

- GitHub Actions mints a short-lived JWT
- Snowflake validates the token's issuer + subject claim
- Two service users with environment-based subjects:
  - `SVC_GITHUB_ACTIONS_STAGE` -- subject: `repo:owner/repo:environment:STAGE`
  - `SVC_GITHUB_ACTIONS` -- subject: `repo:owner/repo:environment:PROD`

### Network Access

GitHub runner IPs are allowed via Snowflake's managed network rule:
```
SNOWFLAKE.NETWORK_SECURITY.GITHUBACTIONS_GLOBAL
```
This is assigned per-user (not account-wide) to minimize blast radius.

### Branch Protection

| Rule | Effect |
|------|--------|
| Require PRs | No direct commits to main |
| Require `lint-and-test` status check | Code must pass lint + tests before merge |
| 1 approving review required | Peer review enforced |
| Dismiss stale reviews | New commits invalidate prior approvals |
| No force pushes | History cannot be rewritten |
| PROD environment reviewer | Manual prod deploy requires explicit approval |

### Deployment Flow

```
Developer pushes code
        │
        ▼
┌──────────────────────┐
│   PR to main         │
│   - ruff lint        │
│   - ruff format      │
│   - pytest           │
└──────────┬───────────┘
           │ (merge)
           ▼
┌──────────────────────┐
│   Push to main       │
│   (auto-trigger)     │
│                      │
│   STAGE Deploy:      │
│   1. Train model     │
│   2. Register        │
│   3. Replicate→PROD  │
└──────────┬───────────┘
           │ (success)
           ▼
┌──────────────────────┐
│   Manual Dispatch    │
│   (click Run)        │
│                      │
│   PROD Deploy:       │
│   1. Verify model    │
│   2. Deploy service  │
│   3. Health check    │
└──────────────────────┘
```

---

## Infrastructure

### Snowflake Objects per Environment

| Object | DEV | STAGE | PROD |
|--------|-----|-------|------|
| Database | `SNOW_MLOPS_DEV` | `SNOW_MLOPS_STAGE` | `SNOW_MLOPS_PROD` |
| Schema | `ML` | `ML` | `ML` |
| Warehouse | `SNOW_MLOPS_DEV_WH` | `SNOW_MLOPS_STAGE_WH` | `SNOW_MLOPS_PROD_WH` |
| Compute Pool | `SNOW_MLOPS_DEV_POOL` | `SNOW_MLOPS_STAGE_POOL` | `SNOW_MLOPS_PROD_POOL` |
| Stages | ML_ARTIFACTS, DAG_STAGE, JOB_STAGE | Same | Same |

### GitHub Variables (repo settings)

| Variable | Value |
|----------|-------|
| `SNOWFLAKE_ACCOUNT` | Account identifier |
| `SNOWFLAKE_DATABASE_STAGE` | `SNOW_MLOPS_STAGE` |
| `SNOWFLAKE_DATABASE_PROD` | `SNOW_MLOPS_PROD` |
| `SNOWFLAKE_SCHEMA` | `ML` |

---

## Project Structure

```
snowflake-mlops/
├── .github/workflows/
│   ├── pr-checks.yml            # Lint + format + tests on PRs
│   ├── deploy.yml               # STAGE: train + register + replicate
│   └── deploy-prod.yml          # PROD: deploy inference service
├── scripts/
│   ├── setup.sh                 # Create all Snowflake infra (DEV + PROD)
│   ├── setup_cicd.sh            # Configure OIDC users + network policy
│   ├── setup_stage_features.py  # One-time Feature Store setup for STAGE
│   ├── generate_dataset.py      # Synthetic data generation (both envs)
│   ├── run_training_job.py      # DEV: submit training to compute pool
│   ├── run_stage_pipeline.py    # STAGE: train + register + replicate
│   └── deploy_prod_service.py   # PROD: deploy inference service
├── source/
│   ├── config.py                # Centralized configuration
│   ├── snowpark_session.py      # Session helper (local + CI auth)
│   ├── features/                # Feature Store entities + views
│   ├── training/                # Model training + evaluation
│   ├── registry/                # Model registry operations
│   ├── serving/                 # SPCS service deployment
│   └── pipeline/                # ML Task DAG definition
├── tests/
│   ├── test_config.py           # Config validation (runs in CI)
│   └── test_endpoint.py         # Inference service integration tests
├── Notebooks/                   # Walkthrough notebooks (01-05)
└── docs/                        # This documentation
```

---

## Configuration

All pipeline parameters are centralized in `source/config.py`:

```python
# Target environment (where pipeline WRITES)
DATABASE = "SNOW_MLOPS_DEV"
SCHEMA = "ML"

# Source environment (where raw data LIVES -- always PROD)
SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"

# Pipeline defaults (all strings for DAG config passthrough)
PIPELINE_CONFIG = {
    "database": DATABASE,
    "source_database": SOURCE_DATABASE,
    "n_estimators": "200",
    "learning_rate": "0.1",
    "max_depth": "6",
    "scale_pos_weight": "33",
    "min_auc_roc": "0.85",
    "min_precision": "0.70",
    "min_recall": "0.60",
    "model_name": "MLOPS_FRAUD_DETECTOR",
    "service_name": "MLOPS_FRAUD_DETECTOR_SERVICE",
    "max_instances": "2",
}
```

---

## Setup Instructions

### First-time setup

```bash
# 1. Create Snowflake infrastructure
bash scripts/setup.sh

# 2. Configure CI/CD access (OIDC + network policy)
bash scripts/setup_cicd.sh

# 3. Generate synthetic data (uploads to DEV + PROD)
SNOWFLAKE_CONNECTION_NAME=<your_conn> uv run python scripts/generate_dataset.py

# 4. Setup Feature Store in STAGE
SNOWFLAKE_CONNECTION_NAME=<your_conn> uv run python scripts/setup_stage_features.py
```

### Running the pipeline locally (DEV)

```bash
# Train model on compute pool
SNOWFLAKE_CONNECTION_NAME=<your_conn> uv run python scripts/run_training_job.py

# Test inference endpoint
SNOWFLAKE_CONNECTION_NAME=<your_conn> uv run pytest tests/test_endpoint.py -v
```

### Running the full MLOps flow

1. Make changes on a feature branch
2. Open PR to `main` -- lint/format/tests run automatically
3. Merge PR -- STAGE deploys (train, register, replicate)
4. Manually trigger PROD deploy from Actions tab
5. PROD service serves the new model version

---

## Key Design Decisions

1. **Environment separation by database** -- Same account, different databases. Simple RBAC, no cross-account replication needed.

2. **Source data always in PROD** -- Single source of truth. DEV and STAGE never own raw data, only derived artifacts.

3. **No inference service in STAGE** -- STAGE validates training + registration. Service deployment is tested only in DEV (full cycle) and PROD (actual serving).

4. **Model replication via SQL** -- `CREATE MODEL...FROM MODEL` is atomic and preserves all metadata. No pickle files shipped between environments.

5. **OIDC for CI/CD** -- Zero stored secrets. GitHub Actions tokens are short-lived and environment-scoped.

6. **Manual PROD gate** -- `workflow_dispatch` + environment protection rules prevent accidental production deployments.

7. **`@remote` for training** -- Compute pool execution means training runs at scale inside Snowflake, not on CI runners. GitHub Actions only orchestrates.
