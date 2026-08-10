# Production MLOps on Snowflake: A Git-Native Framework

Machine learning teams face a persistent challenge: how do you safely promote models from experimentation to production? How do you prevent regressions? How do you roll back when predictions fail? This article presents a production-ready MLOps framework built on Snowflake that treats models as first-class infrastructure—versioned, tested, and deployed through automated CI/CD pipelines.

This is **not** a tutorial on training models. This is about the infrastructure and deployment patterns that let you ship ML changes safely and frequently. The framework targets **single-account deployments** with three environments (DEV, STAGE, PROD as separate databases), though the patterns extend naturally to multi-account topologies for enterprises requiring stricter isolation.

---

## The Architecture: Git-Driven Model Promotion

The framework implements a **three-stage promotion pipeline** where every environment maps to a git branch:

```
main branch    → PROD database   (production traffic)
stage branch   → STAGE database  (pre-production validation)  
feature/* branches → DEV database   (rapid iteration)
```

**Core principle**: The state of your production models always matches the state of your `main` branch. Want to know what's deployed? Check git tags. Want to roll back? Revert a commit.

### Deployment Flow

```
┌─────────────────────────────────────────────────────┐
│          Feature Branch → DEV                        │
│  • Push triggers CI                                  │
│  • Basic linting only                                │
│  • Deploy for manual testing                         │
└─────────────────────────────────────────────────────┘
                        ↓
                   PR to stage
                        ↓
┌─────────────────────────────────────────────────────┐
│          Stage Branch → STAGE                        │
│  • Full test suite runs                              │
│  • Train model on STAGE compute pool                 │
│  • Quality gate checks metrics                       │
│  • Batch inference validates predictions             │
│  • Model replicates to PROD registry (if passing)    │
└─────────────────────────────────────────────────────┘
                        ↓
                PR to main + approval
                        ↓
┌─────────────────────────────────────────────────────┐
│          Main Branch → PROD                          │
│  • Requires manual approval                          │
│  • Blue/green SPCS deployment                        │
│  • Batch inference validation                        │
│  • Gateway cutover                                   │
│  • Git release tag                                   │
└─────────────────────────────────────────────────────┘
```

---

## The Three Environments

### DEV: Fast Feedback Loop

**Trigger**: Push to any `feature/*` branch  
**Purpose**: Rapid iteration and manual testing  
**Quality bar**: Lint only (no tests)

```bash
# Developer workflow
git checkout -b feature/fraud-v2
# Edit training code
git push origin feature/fraud-v2
# CI deploys to SNOW_MLOPS_DEV.ML automatically
```

DEV exists for speed. Break things, experiment, test manually. The only gate is basic code formatting.

### STAGE: Pre-Production Validation

**Trigger**: PR merge to `stage` branch  
**Purpose**: Validate models before production  
**Quality bar**: Full test suite + quality gate

STAGE is where models prove they work. The pipeline:

1. **Registers features**: Creates Dynamic Tables in Feature Store
2. **Trains model**: Runs `@remote` training job on dedicated compute pool  
3. **Quality gate**: Compares metrics (AUC-ROC, F1, precision) against thresholds
4. **Replicates model**: If passing, copies model to PROD registry using `ALTER MODEL ... ADD VERSION FROM MODEL`
5. **Batch inference**: Scores test dataset and validates prediction schema

**Key insight**: The model is replicated to PROD registry but not yet serving traffic. PROD still uses the old version until you explicitly deploy.

### PROD: Production Deployment

**Trigger**: PR merge to `main` (with approval)  
**Purpose**: Serve production traffic  
**Quality bar**: STAGE passed + human approval

PROD deployment is a **two-step process**:

1. **Approval gate**: GitHub environment protection requires at least one team member to approve
2. **Blue/green deployment**: New SPCS service version spins up, health checks pass, Gateway switches traffic

If anything fails, the old service continues serving. No downtime, no partial rollout.

---

## Model Registry: The Source of Truth

Models in Snowflake Model Registry are versioned objects with full metadata:

```sql
-- Each version has:
-- • Feature Store snapshot (which features were used)
-- • Training metrics (AUC-ROC, F1, precision)
-- • Git commit SHA (exact code that produced this model)
-- • Target platforms (warehouse, SPCS, or both)
```

### Cross-Database Replication

Models are promoted from STAGE to PROD using atomic SQL:

```sql
-- Copy version V3 from STAGE to PROD
ALTER MODEL SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR
  ADD VERSION V3
  FROM MODEL SNOW_MLOPS_STAGE.ML.MLOPS_FRAUD_DETECTOR VERSION V3;

-- Make it the active version for batch inference
ALTER MODEL SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR
  SET DEFAULT_VERSION = V3;
```

This preserves all metadata, signatures, and configuration. The PROD model is immediately usable for both batch and real-time inference.

### Model Versioning Strategy

**Version naming**: `V{counter}` (V1, V2, V3...) tracked in the model object  
**Git tagging**: After successful PROD deploy → `prod/V11-20260803-151530`  
**Rollback tagging**: After emergency rollback → `prod/V10-rollback-20260803-160000`

To see what's deployed in PROD:

```bash
# From git
git tag -l "prod/*" --sort=-creatordate | head -1

# From Snowflake
SHOW MODELS LIKE 'MLOPS_FRAUD_DETECTOR' IN SNOW_MLOPS_PROD.ML;
-- Check "default_version_name" column
```

---

## The Quality Gate: Preventing Bad Models from Reaching Production

The quality gate is a Python function that compares training metrics against thresholds:

```python
def check_quality_gate(metrics: dict) -> bool:
    """
    Returns True if metrics pass, False if blocked.
    Thresholds are configurable per model.
    """
    thresholds = {
        "auc_roc": 0.85,
        "f1_score": 0.75,
        "precision": 0.80
    }
    
    for metric, min_value in thresholds.items():
        if metrics.get(metric, 0) < min_value:
            print(f"❌ {metric} = {metrics[metric]} below threshold {min_value}")
            return False
    
    return True
```

**What happens when the gate fails:**

- Model is registered in STAGE (you can inspect it)
- Model is **not replicated** to PROD
- CI pipeline stops with clear error message
- You fix the model and re-run

**What happens when the gate passes:**

- Model is replicated to PROD registry
- STAGE completes successfully
- You can now deploy to PROD (requires approval)

---

## The Full Promotion Flow

Here's the complete step-by-step for a STAGE → PROD deployment:

| Step | What Happens | Script / Location | If It Fails |
|------|-------------|-------------------|-------------|
| 1 | Register entities + feature views in STAGE (creates Dynamic Tables) | `run_stage_pipeline.py → register_feature_views()` | Pipeline stops |
| 2 | Submit `@remote` training job to STAGE compute pool | `run_stage_pipeline.py → train_and_register_stage()` | Pipeline stops |
| 3 | Compare metrics against thresholds | `check_quality_gate(metrics)` | Model registered but NOT replicated |
| 4 | `ALTER MODEL ... ADD VERSION FROM MODEL` | `deploy/strategies/single_account.py` | Pipeline stops (PROD doesn't get the model) |
| 5 | Score feature table in STAGE, validate predictions | `run_batch_inference.py` | Pipeline fails (flags bad model) |
| 6 | Reviewer approves PROD deployment (environment gate) | GitHub Actions UI | N/A (manual action) |
| 7 | Create versioned SPCS service → health check → shift gateway | `deploy_prod_service.py` | Old service stays live (no rollback needed) |
| 8 | Score feature table in PROD, validate predictions | `run_batch_inference.py` | Flags issue but service already live |

---

## Pluggable Promotion Strategies

The model promotion step (Step 4) is implemented as a **pluggable strategy** controlled by the `TOPOLOGY` environment variable:

| Topology | Mechanism | When to Use |
|----------|-----------|-------------|
| `single-account` (default) | `ALTER MODEL ... ADD VERSION FROM MODEL` | All environments in one account |
| `multi-account` | Cross-account model sharing | Separate accounts per environment |
| `cross-region` | Replication group | Accounts in different regions |

**Single-account** (this framework):
```
One Snowflake account
├── SNOW_MLOPS_DEV (database)
├── SNOW_MLOPS_STAGE (database)
└── SNOW_MLOPS_PROD (database)
```

**Multi-account** (extension):
```
Three Snowflake accounts
├── dev_account (entire account for DEV)
├── stage_account (entire account for STAGE)
└── prod_account (entire account for PROD)
```

The CI/CD workflow, testing gates, and approval flows remain identical. You're just deploying to different accounts instead of different databases. The model promotion mechanism changes from direct SQL replication to cross-account sharing, but the pipeline structure is unchanged.

---

## CI/CD Infrastructure: A Single Unified Workflow

The entire pipeline is orchestrated by a **single GitHub Actions workflow** (`deploy.yml`) with four jobs arranged as a dependency graph:

```
┌──────┐     ┌──────┐
│ lint │     │ test │
└───┬──┘     └───┬──┘
    └────┬───────┘
         ↓
   ┌─────────────┐
   │ deploy-stage │
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │ deploy-prod  │ (requires approval)
   └─────────────┘
```

### Job Breakdown

| File | Trigger | Environment | Actions |
|------|---------|-------------|---------|
| `deploy.yml → lint job` | Push to main OR manual dispatch | — | Ruff lint + format check |
| `deploy.yml → test job` | Push to main OR manual dispatch | — | Unit tests (pytest) |
| `deploy.yml → deploy-stage job` | After lint + test pass | STAGE | Register features, train, quality gate, replicate, batch inference |
| `deploy.yml → deploy-prod job` | After STAGE passes + reviewer approval | PROD | Blue/green SPCS deploy, batch inference, verify gateway, tag release |
| `rollback.yml` | Manual dispatch (model version + reason) | PROD | Set model version, redeploy SPCS service, validate, tag rollback |
| `pr-checks.yml` | PR opened/updated against main | — | Lint + test (gates merge) |

### STAGE Job Steps

```yaml
name: Deploy

on:
  push:
    branches: [main]
    paths-ignore: ['*.md', 'docs/**', 'LICENSE']
  workflow_dispatch:
    inputs:
      reason:
        description: 'Reason for manual retrain'
        required: false

permissions:
  id-token: write        # Required for OIDC token
  contents: write        # Required for git tags on PROD deploy
```

**Steps executed:**

1. **Checkout** — Clone repo with `fetch-depth: 2`
2. **Snowflake CLI setup** — `snowflakedb/snowflake-actions@v3` with `use-oidc: true`
3. **Connection test** — `snow connection test -x` validates auth before expensive steps
4. **Install uv + Python** — Fast package manager, deterministic installs
5. **Install deps** — `uv sync --group snow --no-default-groups` (lean: only orchestration packages)
6. **Run STAGE pipeline** — `scripts/run_stage_pipeline.py` (train → gate → replicate)
7. **Run batch inference** — `scripts/run_batch_inference.py` (score + validate)

### PROD Job Steps

1. **Verify model exists** — Confirm the replicated model is in PROD registry
2. **Deploy SPCS service** — Blue/green deployment with health check
3. **Register Feature Views** — Ensure Dynamic Tables exist in PROD
4. **Run batch inference** — Score and validate predictions
5. **Verify gateway** — Confirm traffic routes to new service
6. **Tag release** — Create git tag `prod/<VERSION>-<TIMESTAMP>` marking what's deployed

---

## Authentication: OIDC, No Stored Secrets

GitHub Actions authenticates to Snowflake using **OIDC workload identity**. GitHub mints a short-lived JWT; Snowflake validates the issuer and subject claim. No passwords or API keys are stored anywhere.

```
┌─────────────────┐
│ GitHub Actions  │
└────────┬────────┘
         │ 1. Request OIDC token
         ↓
┌─────────────────┐
│ GitHub OIDC     │
│ Provider        │
└────────┬────────┘
         │ 2. Mint JWT (subject: repo:owner@ID/repo@ID:environment:PROD)
         ↓
┌─────────────────┐
│ Snowflake       │
│ Account         │
└────────┬────────┘
         │ 3. Validate issuer + subject
         │ 4. Return session token
         ↓
   CI pipeline runs
```

### Service User Configuration

Each GitHub environment has its own service user:

```sql
CREATE USER SVC_GITHUB_ACTIONS_STAGE
  TYPE = SERVICE
  DEFAULT_ROLE = MLOPS_DEPLOY_ROLE
  WORKLOAD_IDENTITY = (
    TYPE = OIDC
    ISSUER = 'https://token.actions.githubusercontent.com'
    SUBJECT = 'repo:owner@ID/repo@ID:environment:STAGE'
  );

ALTER USER SVC_GITHUB_ACTIONS_STAGE 
  SET NETWORK_POLICY = 'GITHUB_ACTIONS_POLICY';
```

### Environment Variables Set by Snowflake Action

The `snowflake-actions@v3` action automatically sets these after successful OIDC exchange:

| Variable | Value | Purpose |
|----------|-------|---------|
| `SNOWFLAKE_TOKEN` | (short-lived token) | Session authentication |
| `SNOWFLAKE_AUTHENTICATOR` | `WORKLOAD_IDENTITY` | Tells connector which auth method |
| `SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER` | `OIDC` | Required for workload identity flow |
| `SNOWFLAKE_AUDIENCE` | `snowflakecomputing.com` | Token audience validation |

### Session Helper

The project includes a helper that detects whether it's running in CI or locally:

```python
def create_snowpark_session():
    # CI/OIDC: detected by SNOWFLAKE_TOKEN + SNOWFLAKE_ACCOUNT in env
    if os.getenv("SNOWFLAKE_TOKEN") and os.getenv("SNOWFLAKE_ACCOUNT"):
        config = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "token": os.environ["SNOWFLAKE_TOKEN"],
            "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "oauth"),
        }
        # Pass OIDC-specific params
        if os.getenv("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"):
            config["workload_identity_provider"] = os.environ["SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"]
        if os.getenv("SNOWFLAKE_AUDIENCE"):
            config["audience"] = os.environ["SNOWFLAKE_AUDIENCE"]
        return Session.builder.configs(config).create()

    # Local dev: use connections.toml
    connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    return Session.builder.configs({"connection_name": connection_name}).create()
```

**Critical detail**: The authenticator is `WORKLOAD_IDENTITY`, not `oauth`. If you hardcode `"oauth"`, CI fails with "Invalid OAuth access token." Always read it from the environment variable.

---

## Git Branching and Release Tagging

The project uses **trunk-based development** with feature branches:

```
feature/* branch → PR → main → deploy.yml → prod/V11-20260803
```

**Branching rules:**

- All development happens on `feature/*` or `fix/*` branches
- `main` is protected: merges require passing lint + test checks + 1 reviewer approval
- Enforce for admins: even repository admins must go through PRs (no direct pushes)

**Release tags:**

- After successful PROD deploy: `prod/V11-20260803-151530`
- After rollback: `prod/V10-rollback-20260803-160000`

To see what's currently deployed:

```bash
git tag -l "prod/*" --sort=-creatordate | head -1
```

---

## Rollback: When Things Go Wrong

The rollback workflow (`rollback.yml`) provides emergency reversion to a previous model version. Triggered via manual dispatch with two required inputs:

- `model_version` — The version to roll back to (e.g., V10)
- `reason` — Audit trail for why the rollback was performed

**What the rollback workflow does:**

1. Sets `DEFAULT_VERSION` on the model (batch inference uses old model immediately)
2. Deploys the SPCS service for the rolled-back version (blue/green)
3. Validates batch inference still works
4. Creates a rollback tag: `prod/V10-rollback-20260803-160000`

**Key characteristic**: Rollback is as automated as forward deployment. No manual SQL. No scrambling through logs. You specify a version number and hit "Run workflow."

---

## RBAC and Security

CI/CD runs with a **least-privilege service role** (`MLOPS_DEPLOY_ROLE`) — never ACCOUNTADMIN. This role has exactly the permissions needed to run the pipeline and nothing more.

### MLOPS_DEPLOY_ROLE Permissions

| Category | Grants |
|----------|--------|
| Database access | `USAGE` on all three databases (DEV, STAGE, PROD) |
| Schema control | `ALL` on `.ML` schema in each database |
| Objects | `ALL` on all tables, dynamic tables, stages, models in each schema |
| Compute | `USAGE` on all warehouses and compute pools |
| Account-level | `BIND SERVICE ENDPOINT`, `EXECUTE TASK`, `EXECUTE MANAGED TASK` |

### Model Ownership

Models are Snowflake objects with ownership-based access control. A role can only modify models it owns. The setup script transfers ownership of any existing models to `MLOPS_DEPLOY_ROLE`:

```sql
-- Transfer existing models (created by ACCOUNTADMIN during initial setup)
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_DEV.ML 
  TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_STAGE.ML 
  TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_PROD.ML 
  TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
```

**Why this matters**: Without this ownership transfer, CI fails with "Model 'MLOPS_FRAUD_DETECTOR' already exists, but current role has no privileges on it." This happens because `setup.sh` creates infrastructure as ACCOUNTADMIN, but CI runs as `MLOPS_DEPLOY_ROLE`.

### Network Policy

GitHub Actions runners use dynamic IPs. Snowflake provides a managed network rule that automatically tracks them:

```sql
CREATE NETWORK POLICY GITHUB_ACTIONS_POLICY
  ALLOWED_NETWORK_RULE_LIST = ('SNOWFLAKE.NETWORK_SECURITY.GITHUBACTIONS_GLOBAL');
```

This is applied to service users only — it doesn't affect human users or other integrations. Even if someone steals the OIDC configuration, they can't authenticate from outside GitHub Actions.

---

## Lean CI Installs

The project uses `uv` dependency groups to separate CI from dev dependencies:

```toml
# pyproject.toml
[dependency-groups]
snow = [
    "snowflake-snowpark-python>=1.53.1",
    "snowflake-ml-python>=1.47.0",
]
dev = ["ruff", "pytest"]
```

CI runs: `uv sync --group snow --no-default-groups` — installs only the Snowflake packages needed for orchestration. Dev tools (ruff, pytest) are not installed in the deploy workflow. This keeps CI fast and reduces attack surface.

---

## GitHub Repository Configuration

Required settings for the framework to work:

| Setting | Where | Value |
|---------|-------|-------|
| `SNOWFLAKE_ACCOUNT` | Repository variable | Your account identifier (e.g., `MYORG-MYACCOUNT`) |
| `SNOWFLAKE_DATABASE_STAGE` | Repository variable | `SNOW_MLOPS_STAGE` |
| `SNOWFLAKE_DATABASE_PROD` | Repository variable | `SNOW_MLOPS_PROD` |
| `SNOWFLAKE_SCHEMA` | Repository variable | `ML` |
| `SNOWFLAKE_USER_STAGE` | Repository variable | `SVC_GITHUB_ACTIONS_STAGE` |
| `SNOWFLAKE_USER_PROD` | Repository variable | `SVC_GITHUB_ACTIONS` |
| `TOPOLOGY` | Repository variable | `single-account` |
| STAGE environment | Settings → Environments | No protection rules |
| PROD environment | Settings → Environments | Required reviewers (approval gate) |
| Branch protection | Settings → Branches → main | Require PR, require `lint` + `test` checks, enforce for admins |

---

## Real-World Workflow: Deploying a Fraud Detection Model

Here's what it looks like in practice:

### Step 1: Develop on a Feature Branch

```bash
git checkout -b feature/fraud-v2
# Edit src/train_fraud_model.py
git commit -am "Improve fraud detection with velocity features"
git push origin feature/fraud-v2
```

CI automatically deploys to DEV. You test manually in `SNOW_MLOPS_DEV.ML.MLOPS_FRAUD_DETECTOR`.

### Step 2: Promote to STAGE

```bash
gh pr create --base stage --title "Fraud detector v2"
# CI runs: lint, test, train, quality gate, replicate
# If metrics pass, model is replicated to PROD registry
git checkout stage
git merge feature/fraud-v2
git push
```

You validate in STAGE:
- Check training metrics in CI logs
- Run manual batch inference tests
- Review predictions for sanity

### Step 3: Deploy to PROD

```bash
gh pr create --base main --title "Deploy fraud detector v2 to PROD"
# PR requires 1 reviewer approval
# After approval + merge:
# - Blue/green SPCS deploy
# - Batch inference validation
# - Gateway cutover
# - Git tag: prod/V11-20260803-151530
```

Your new fraud detector is live.

### Step 4: Rollback (If Needed)

Production predictions look wrong? Rollback is a manual workflow dispatch:

```bash
# Go to Actions → Rollback workflow → Run workflow
# Inputs:
#   model_version: V10
#   reason: "High false positive rate on corporate cards"
# CI executes:
# - SET DEFAULT_VERSION = V10
# - Blue/green redeploy of V10 service
# - Validation
# - Git tag: prod/V10-rollback-20260803-160000
```

Rollback completes in minutes. Your system is back to the previous version.

---

## Key Takeaways for ML Engineers

### What This Framework Gives You

1. **Versioned models**: Every model version is tracked with git commit SHA, training metrics, and feature snapshot
2. **Automated quality gates**: Bad models never reach production
3. **Environment isolation**: Break things in DEV without affecting users
4. **Approval gates**: Production changes require human review
5. **Instant rollbacks**: Revert to any previous version in minutes
6. **Audit trail**: Git tags show exactly what's deployed and when
7. **Least-privilege security**: CI runs with minimal permissions
8. **No stored secrets**: OIDC authentication means no passwords to rotate

### What This Framework Does NOT Cover

This is an **infrastructure and deployment framework**. It assumes you already know how to:

- Write training code with Snowpark ML
- Define features in Feature Store
- Build inference pipelines (batch or real-time)
- Monitor model performance

The framework focuses on **safely promoting models through environments**, not on the ML engineering itself.

### Single-Account vs Multi-Account

This framework targets **single-account deployments**:

```
One account
├── SNOW_MLOPS_DEV (database)
├── SNOW_MLOPS_STAGE (database)
└── SNOW_MLOPS_PROD (database)
```

**Pros**: Simpler setup, lower cost, easier data sharing, no cross-account networking  
**Cons**: Less isolation, shared compute quotas, single blast radius

**To extend to multi-account**: Change the topology strategy from `single-account` to `multi-account`. The CI/CD workflow, testing gates, and approval flows remain identical. You're just deploying to different accounts instead of different databases. Model promotion changes from direct SQL replication to cross-account sharing (via Snowflake's built-in sharing mechanisms), but the pipeline structure is unchanged.

---

## Common Pitfalls

### Pitfall 1: Models Breaking in PROD but Not STAGE

**Cause**: Data differences between environments

**Solution**: Use production data samples in STAGE:

```sql
CREATE TABLE SNOW_MLOPS_STAGE.SAMPLE_TRANSACTIONS 
CLONE SNOW_MLOPS_PROD.TRANSACTIONS;

CREATE TASK refresh_stage_sample
  WAREHOUSE = ML_WH
  SCHEDULE = 'DAILY'
AS
  CREATE OR REPLACE TABLE SNOW_MLOPS_STAGE.SAMPLE_TRANSACTIONS 
  AS SELECT * FROM SNOW_MLOPS_PROD.TRANSACTIONS SAMPLE (10 ROWS);
```

### Pitfall 2: Manual Drift Between Environments

**Cause**: Someone manually modifies PROD objects

**Solution**: Use ownership-based access control. Only `MLOPS_DEPLOY_ROLE` owns models. Human users can query models but can't modify them. All changes go through CI/CD.

### Pitfall 3: "Model Already Exists" Error in CI

**Cause**: `setup.sh` creates objects as ACCOUNTADMIN, but CI runs as `MLOPS_DEPLOY_ROLE`

**Error message**: "Model 'MLOPS_FRAUD_DETECTOR' already exists, but current role has no privileges on it"

**Solution**: Transfer ownership after initial setup (shown in RBAC section above)

---

## Conclusion

Production MLOps is not about training better models—it's about building **reliable, auditable, and reversible deployment pipelines**. This framework gives you:

- Git-driven deployments: every model version is tracked and reversible
- Automated testing gates: models don't reach production without proving they work
- Environment isolation: break things in DEV without affecting users
- Approval gates for production: humans review before deployment
- Instant rollbacks: revert in minutes, not hours

The patterns here work for single-account deployments. For enterprises requiring stricter isolation, the same patterns extend to multi-account architectures—just deploy to separate accounts instead of separate databases.

Models are too important to deploy manually. Build a pipeline that gives you confidence to ship ML changes frequently and safely.

---

## Additional Resources

- [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)
- [Snowflake ML Jobs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/overview)
- [Snowflake Model Registry](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)
- [Warehouse Inference](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/warehouse)
- [Real-Time Inference (REST API)](https://docs.snowflake.com/en/developer-guide/snowflake-ml/inference/real-time-inference-rest-api)
- [Snowflake Gateway](https://docs.snowflake.com/en/developer-guide/snowflake-ml/inference/stable-endpoints-api-reference)
- [Snowflake CLI GitHub Action](https://docs.snowflake.com/en/developer-guide/snowflake-cli/cicd/github-action)
