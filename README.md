# Snowflake MLOps Template

An open-source, production-ready template for building end-to-end ML pipelines on Snowflake. This framework provides data science teams with starter code they can plug their own model logic into, while giving ML engineering teams a standardized process for training, validating, deploying, and monitoring models — all orchestrated through Git workflows with human-in-the-loop approvals.

Use this template as a bootstrap for your MLOps workflows, or as a baseline to extend with [Cortex Code](https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code) for more customized agentic ML and MLOps workflows.

## What This Template Covers

- **Multi-step ML Pipelines** — Snowflake Task DAG (Python SDK) with configurable compute per step (warehouse or SPCS compute pools)
- **Feature Engineering** — Snowflake Feature Store with Dynamic Tables and configurable refresh intervals
- **Model Training** — ML Jobs (`@remote`) on compute pools with experiment tracking; registered straight from the pipeline with lineage to the training Dataset
- **Quality Gates** — Configurable metric bars plus champion/challenger comparison against the model in PROD; rejected candidates are removed from the registry
- **Model Versioning** — Auto-incremented versions in the Snowflake Model Registry with metrics, Dataset, and experiment lineage
- **Data Checks** — Row-count, null-rate, and freshness checks on source tables before training
- **CI/CD Workflows** — GitHub Actions with OIDC auth, PR checks, and environment-gated approvals
- **Model Promotion** — Two paths: code-driven (push to main) and data-driven (scheduled retrain with candidate review)
- **Batch Inference** — Scheduled Task scoring with Model Registry `run()`; predictions appended with model version for full history
- **Real-Time Inference** — SPCS inference services with canary Gateway deployment, parity health checks, and git-driven traffic management
- **Serving Modes** — Batch, real-time, or both, chosen in config; CI skips what you don't use
- **Model Monitoring** — Snowflake ML Observability for drift *and* performance (ground-truth join), automatic baseline, drift alerts
- **Rollback** — One-click revert to any previous model version
- **Scheduled Retraining** — Weekly cron with GitHub Issue notifications for human review

## How It Works

**Code-driven promotion:** feature branch → PR (code quality checks) → merge to main → deploy-stage (train, gate, register, validate) → human approval → deploy-prod (promote, batch schedule and/or real-time canary, monitor, tag)

**Data-driven promotion:** weekly cron retrains on fresh data → registers candidate in STAGE → creates GitHub Issue → human reviews → manual promote to PROD

## For Data Science Teams

Swap in your own model — the infrastructure stays the same:

1. **Describe your model** under `model:` in `source/configs/config.yml` — `task_type` (`binary_classification`, `multiclass`, `regression`), `predict_function`, `id_columns`, `label_column`
2. **Replace the two example functions** in `source/model_training/training.py`: `build_dataset(session)` and `train_model(X_train, y_train)`. Any model the Model Registry supports works (scikit-learn, XGBoost, LightGBM, PyTorch, ...).
3. **Update feature engineering** in `source/feature_engineering/feature_views.py`
4. **Set your bar** under `quality_gate:` — metric rules for your task type and the champion comparison

Registration, the quality gate, batch scoring, real-time health checks, and monitoring read the model's signature and these settings — they contain no model-specific code.

Each pipeline step is a self-contained package under `source/` with a logic module exposing `run(session) -> dict` and an `entrypoint.py` CLI. You can run any step on its own without deploying the DAG (see [Run a single step](#run-a-single-step)).

Everything else (CI/CD, versioning, deployment, monitoring) works automatically.

## For ML Engineering Teams

This template provides:

- Standardized ML pipeline structure (Task DAG with configurable compute)
- Automated CI/CD with model approval gates (no model goes to PROD without human review)
- Environment isolation (DEV → STAGE → PROD as separate databases)
- Reproducible deployments (every PROD release tagged, every model version tracked)
- Rollback capability (one-click revert via GitHub Actions)

## Pipeline Flow

### Code-Driven Promotion (push to main)

1. **Code Quality** — Ruff linting, format checks, and unit tests run on every PR and push
2. **Merge to main** — Triggers the deploy workflow
3. **Deploy Task DAG** — Creates/updates Snowflake Tasks using the Python SDK
4. **Execute Pipeline** — Runs three steps sequentially, each on its configured compute (warehouse or SPCS compute pool):
   - Feature Engineering (`source/feature_engineering/` — runs data checks on source tables, then registers the Feature Views)
   - Model Training (`source/model_training/` — trains on a grouped holdout split, scores the PROD champion on the same holdout, logs the run to Experiment Tracking, and registers the candidate in the STAGE registry with lineage to the training Dataset)
   - Evaluation (`source/evaluate/` — the **quality gate**: metric rules plus champion/challenger. A rejected candidate is removed from the registry and the DAG fails.)
5. **Read Candidate** — CI publishes the gate decision (candidate vs champion metrics) to the Job Summary and passes the gated version to downstream jobs. Nothing touches PROD yet.
6. **Batch Inference Validation** — Scores the batch input with the *candidate* version into `BATCH_PREDICTIONS_VALIDATION` and validates outputs against the model signature (no nulls; probabilities sum to 1 for classifiers).
7. **Monitor Validation** — Creates a ModelMonitor for the candidate in STAGE, verifies it reaches ACTIVE state, then drops it (proves PROD setup will succeed)
8. **Human Approval** — Workflow pauses. Reviewer sees candidate and champion metrics in the Job Summary and approves PROD deployment.
9. **PROD Deployment** (steps for disabled serving modes are skipped):
    - **Promote model** — replicates the exact gated version from STAGE to PROD (no re-resolution), sets as DEFAULT
    - **Batch** — validates `model.run()` in PROD, then deploys the scheduled scoring Task (`serving.batch.schedule`) that appends to `BATCH_PREDICTIONS`
    - **Real-time** — deploys a versioned inference service, health-checks it on real rows (valid outputs, parity with warehouse `run()`, agreement with the incumbent), and routes a canary share of Gateway traffic to it. Services that receive no gateway traffic are dropped.
    - **REST smoke test** — real HTTPS calls through the gateway and to each service behind it, authenticated with a temporary PAT; outputs must be valid and match warehouse scoring
    - Sets up the persistent ModelMonitor (drift + performance, automatic baseline, drift alert)
    - **Retention** — prunes old model and Dataset versions (keeps the newest N and anything in use)
    - Tags release: `prod/V3-20260812-...`
    - **Automatic rollback** — if any step fails after promotion (service, health check, gateway, smoke test, monitor), PROD's previous default version and gateway split are restored

### Data-Driven Promotion (scheduled retrain)

1. **Weekly cron fires** (Monday 6AM PT) — or manually triggered
2. **Same pipeline runs** — Feature Eng → Train → Evaluate on fresh data
3. **STAGE_ONLY mode** — Model is registered in STAGE but NOT promoted to PROD
4. **GitHub Issue created** — "Model Candidate Ready: V4" with metrics table
5. **Human reviews** the issue, decides whether to promote
6. **Manual promote** — Run the deploy workflow with `promote_only=true` and specify the `model_version` (e.g., `V4`), approve PROD deployment

### Rollback

Manual dispatch of the rollback workflow: sets the DEFAULT version back, redeploys that version's service if it was dropped, routes 100% of Gateway traffic to it (if real-time is enabled), runs the REST smoke test, and re-validates batch scoring. The scheduled batch Task always uses the default version, so it follows the rollback automatically.

## Snowflake Services Used

| Component | Service |
|-----------|---------|
| Pipeline Orchestration | Tasks (Python SDK DAG) |
| Feature Engineering | Feature Store (Dynamic Tables) |
| Model Training | ML Jobs (`@remote` on Compute Pools) |
| Experiment Tracking | ExperimentTracking API |
| Model Versioning | Model Registry |
| Batch Inference | Scheduled Task + Model Registry `run()` on warehouse |
| Real-Time Serving | SPCS containers + Gateway |
| Model Monitoring | ML Observability (ModelMonitor) |
| CI/CD | GitHub Actions with OIDC (zero secrets) |

## Prerequisites

- Snowflake account with `ACCOUNTADMIN` role (for initial setup; CI uses `MLOPS_STAGE_ROLE` and `MLOPS_PROD_ROLE`)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/) installed
- [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) (`snow`) installed and configured
- [GitHub CLI](https://cli.github.com/) (`gh`) installed
- GitHub repository (public, for environment protection rules)

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/<your-org>/snowflake-mlops.git
cd snowflake-mlops
uv sync
```

### 2. Create Snowflake Infrastructure

```bash
bash scripts/setup.sh
```

Creates three environments (`SNOW_MLOPS_DEV`, `SNOW_MLOPS_STAGE`, `SNOW_MLOPS_PROD`) with databases, warehouses, and compute pools, plus `MLOPS_PYPI_ACCESS_INTEGRATION` (PyPI egress so ML Jobs install the pinned library versions).

### 3. Generate Sample Data

```bash
uv run python scripts/generate_dataset.py
```

Creates 100K synthetic fraud transactions (~3% fraud rate) in `SNOW_MLOPS_PROD.ML`.

### 4. Deploy and Run the Pipeline (DEV)

```bash
# Deploy the Task DAG
uv run python source/pipeline/train_dag.py --deploy --env dev

# Execute (triggers Feature Eng → Training → Evaluation)
uv run python source/pipeline/train_dag.py --execute --env dev

# Check status
uv run python source/pipeline/train_dag.py --status --env dev

# Batch scoring: validate a specific version, or deploy/run the scheduled Task
MODEL_VERSION=V3 uv run python source/pipeline/batch_inference_dag.py --validate --env dev
uv run python source/pipeline/batch_inference_dag.py --deploy --env dev
uv run python source/pipeline/batch_inference_dag.py --execute --env dev
```

`--env` accepts `dev`, `stage`, or `prod`.

#### Run a single step

Each step package ships a CLI entrypoint that calls the exact same `run(session)` the DAG invokes, so local behavior matches the pipeline. Useful for iterating on one step without redeploying the DAG. The target environment comes from `SNOWFLAKE_DATABASE` / `SNOWFLAKE_SCHEMA` / `SNOWFLAKE_WAREHOUSE` (default: DEV):

```bash
uv run python source/feature_engineering/entrypoint.py
uv run --group ml-runtime python source/model_training/entrypoint.py
uv run python source/evaluate/entrypoint.py
```

### 5. Set Up CI/CD

```bash
# Your repo's numeric IDs: gh api repos/<owner>/<repo> --jq '.id, .owner.id'
GITHUB_OWNER=<owner> GITHUB_OWNER_ID=<owner_id> GITHUB_REPO=<repo> GITHUB_REPO_ID=<repo_id> \
  bash scripts/setup_cicd.sh
```

The IDs form the OIDC subject (`repo:<owner>@<owner_id>/<repo>@<repo_id>:environment:<ENV>`) that each service user trusts. The script is idempotent; re-run it after changing grants.

This creates OIDC service users (`SVC_GITHUB_ACTIONS_STAGE`, `SVC_GITHUB_ACTIONS`) for passwordless CI authentication, plus these roles:
- **`MLOPS_STAGE_ROLE`** — used by STAGE workflows (DEV + STAGE access; read-only on PROD source data and PROD models, for the champion comparison)
- **`MLOPS_PROD_ROLE`** — used by PROD workflows (PROD access, BIND SERVICE ENDPOINT)
- **`MLOPS_STAGE_MODEL_OWNER`** — owns the STAGE registry models and is granted to both roles above. Promotion copies a version with `CREATE/ALTER MODEL ... FROM MODEL`, which requires OWNERSHIP on the source model (USAGE only allows inference). This role gives PROD exactly that without inheriting the rest of `MLOPS_STAGE_ROLE`.

Then configure your GitHub repo (**Settings → Secrets and variables → Actions → Variables tab**):

| Variable | Description | Example |
|----------|-------------|---------|
| `SNOWFLAKE_ACCOUNT` | Your Snowflake account identifier | `MYORG-MYACCOUNT` |
| `SNOWFLAKE_DATABASE_STAGE` | STAGE database name | `SNOW_MLOPS_STAGE` |
| `SNOWFLAKE_DATABASE_PROD` | PROD database name | `SNOW_MLOPS_PROD` |
| `SNOWFLAKE_SCHEMA` | Schema (shared across envs) | `ML` |
| `SNOWFLAKE_USER_STAGE` | OIDC service user for STAGE | `SVC_GITHUB_ACTIONS_STAGE` |
| `SNOWFLAKE_USER_PROD` | OIDC service user for PROD | `SVC_GITHUB_ACTIONS` |
| `ENABLE_MODEL_MONITOR` | Set up model monitoring (STAGE validates, PROD creates) | `true` |

Then create **GitHub Environments** (**Settings → Environments**):
- **`STAGE`** — no protection rules
- **`PROD`** — add a required reviewer (this creates the human approval gate)

#### Branch Protection (optional)

The `setup_cicd.sh` script also configures branch protection rules on `main` and creates the GitHub Environments automatically (via `gh` CLI). If your organization manages repository settings separately (e.g., through a platform team, Terraform, or org-level rulesets), you can skip this part of the script and apply equivalent policies through your own process.

What the script configures:

| Setting | Value |
|---------|-------|
| Require pull request before merging | Yes (no direct pushes to `main`) |
| Required approving reviews | 1 |
| Dismiss stale reviews on new pushes | Yes |
| Required status checks | `Code Quality` (must pass before merge) |
| Require branches to be up to date | Yes |
| Enforce for administrators | No (admins can bypass for testing; enable for stricter governance) |
| Allow force pushes | No |
| Allow branch deletion | No |

It also creates two GitHub Environments:
- **`STAGE`** — no protection rules (deploys automatically after merge)
- **`PROD`** — requires reviewer approval (the script adds the authenticated `gh` user as the required reviewer)

These protections ensure every change to `main` goes through: **feature branch → PR → code quality checks pass → review approval → merge → pipeline runs**. This is what prevents untested code from reaching STAGE or PROD.

To apply manually instead of using the script, configure these under **Settings → Branches → Branch protection rules** and **Settings → Environments** in your GitHub repository.

### 6. Test End-to-End

```bash
# Create a feature branch, make a change, push
git checkout -b feature/my-change
# ... modify model logic, features, or config ...
git add -A && git commit -m "Update model" && git push -u origin feature/my-change

# Create PR → code quality checks → merge → deploy-stage → approve → deploy-prod
gh pr create --title "My model update"
```

## Project Structure

```
snowflake-mlops/
├── .github/workflows/
│   ├── deploy.yml                 # PR checks + deploy (code quality → stage → prod)
│   ├── scheduled-retrain.yml      # Cron: retrain → STAGE candidate → notify
│   ├── traffic-shift.yml          # Gateway traffic allocation (gateway-config.yml)
│   └── rollback.yml               # Manual: revert to previous version
├── deploy/
│   ├── promote.py                 # STAGE → PROD model copy (FROM MODEL), sets default
│   └── set_default_version.py     # Used by the rollback workflow
├── scripts/
│   ├── setup.sh                   # Infrastructure provisioning (incl. PyPI integration)
│   ├── setup_cicd.sh              # OIDC users + network policy
│   ├── generate_dataset.py        # Synthetic data generation
│   ├── apply_gateway_config.py    # Gateway traffic split; drops services with no traffic
│   ├── deploy_prod_service.py     # PROD canary SPCS deployment
│   ├── smoke_test_endpoint.py     # Post-deploy HTTPS REST smoke test
│   └── prod_state.py              # Capture/restore PROD state (automatic rollback)
├── source/
│   ├── configs/config.yml         # Pipeline configuration (single source of truth)
│   ├── config.py                  # YAML loader + env resolution (resolve_env_config)
│   ├── snowpark_session.py        # Session helper (SSO + OIDC)
│   ├── feature_engineering/       # Step: feature engineering
│   │   ├── entrypoint.py         # CLI — calls feature_engineering.run(session)
│   │   ├── feature_engineering.py
│   │   └── feature_views.py      # Feature View definitions (canonical schema)
│   ├── model_training/            # Step: model training
│   │   ├── entrypoint.py
│   │   └── training.py
│   ├── evaluate/                  # Step: evaluation
│   │   ├── entrypoint.py
│   │   └── evaluation.py
│   ├── pipeline/
│   │   ├── train_dag.py           # Training Task DAG (per-step warehouse/SPCS)
│   │   └── batch_inference_dag.py # Scheduled batch scoring Task + CI validation
│   ├── serving/
│   │   ├── batch_inference.py     # Score + append predictions (Task entrypoint)
│   │   └── signature.py           # Signature-driven casting + output validation
│   ├── monitoring/
│   │   └── model_monitor.py       # Monitor source/baseline/alert management
│   └── utils/                     # CI helpers
│       ├── wait_for_task.py           # Poll Task DAG completion, export gate result
│       ├── read_candidate.py          # Publish gate decision + gated version to CI
│       ├── serving_flags.py           # Serving modes from config → GITHUB_OUTPUT
│       ├── setup_model_monitor.py     # Model monitoring setup/validation
│       ├── retention.py               # Prune old model / Dataset versions
│       └── notify_candidate.py        # GitHub Issue notification for candidates
├── gateway-config.yml             # Real-time traffic split (edit + merge to shift)
├── tests/                         # Unit tests; test_endpoint.py = live REST tests against PROD
├── notebooks/                     # Educational Jupyter notebooks (01-06; 06 = Feature Store + REST inference)
└── docs/
    └── docs.html                  # Detailed architecture documentation
```

## Configuration

Everything is configurable from `source/configs/config.yml`. The sections you will touch most:

```yaml
# What the model is — serving, validation, and monitoring read this
model:
  name: MLOPS_FRAUD_DETECTOR
  task_type: binary_classification   # binary_classification | multiclass | regression
  predict_function: predict_proba    # predict for regressors
  id_columns: [CUSTOMER_ID]
  label_column: IS_FRAUD

# Quality gate: absolute bars + champion/challenger vs PROD
quality_gate:
  metrics:
    auc_roc: {min: 0.85}
    recall: {min: 0.60}
  champion:
    enabled: true
    source_env: prod
    metric: auc_roc
    higher_is_better: true
    min_improvement: 0.0

# Serving: batch, real-time, or both
serving:
  batch:
    enabled: true
    schedule: "USING CRON 0 7 * * * America/Los_Angeles"
    output_table: BATCH_PREDICTIONS   # appended with MODEL_VERSION / SCORED_AT / RUN_ID
  realtime:
    enabled: true
    gateway_name: FRAUD_DETECTOR_GATEWAY
    min_instances: 1
    max_instances: 2
    canary_weight: 20

# Pre-training data checks on source tables
data_checks:
  RAW_TRANSACTIONS:
    min_rows: 1000
    max_null_pct: {CUSTOMER_ID: 0}
    freshness_column: TIMESTAMP
    max_age_hours: 26

# Monitoring: ground truth enables performance metrics; baseline enables drift
monitor:
  actuals:
    table: RAW_TRANSACTIONS
    join_keys: [CUSTOMER_ID]
    label_sql: "MAX(IS_FRAUD::INT)"
  baseline: auto
  notification_integration: null      # set to email on drift

# Compute mode per pipeline step — "warehouse" or "spcs"
pipeline_steps:
  feature_eng: {compute: spcs}
  train_model: {compute: spcs}
  evaluate: {compute: warehouse}
```

`source/config.py` loads this file and re-exports the values as Python constants (`MODEL_NAME`, `GATE_RULES`, `TRAINING_PARAMS`, etc.), so modules keep a single import surface.

**Compute selection is config-only.** Both compute paths execute the same `run(session)` function in each step's logic module (e.g. `source/model_training/training.py`) — SPCS as an ML Job on a compute pool, warehouse as an anonymous stored procedure. Switching a step between them never requires a code change. Per-environment infrastructure names live under `environments:` in the same file. At deploy time, `train_dag.py` packages `config.py` and `configs/config.yml` into `source.zip` with the step code, and pins the target environment's database/schema/warehouse before each step imports its config.

### ML Runtime Dependencies

Every library version is pinned once, in `pyproject.toml` under the `[dependency-groups] ml-runtime` group:

```toml
ml-runtime = ["snowflake-ml-python==2.4.0", "xgboost==3.3.0", "scikit-learn==1.8.0"]
```

The same pins are used by the CI/local client (`uv.lock`), ML Jobs on SPCS (pip from PyPI), warehouse stored procedures (Snowflake Anaconda channel), and Model Registry serving (`log_model(conda_dependencies=...)`, which records `snowflake-ml-python` at the logging client's version). Every pipeline step checks the installed versions against the pins before it runs and fails if they differ.

ML Jobs reach PyPI through the external access integration named by `pypi_access_integration` in `config.yml` (created by `scripts/setup.sh`, granted to `MLOPS_STAGE_ROLE` by `setup_cicd.sh`). Without it, pip can't reach PyPI and the job silently keeps the runtime image's preinstalled versions; the version check turns that into a clear failure.

To update:
1. Pick versions available on both PyPI and the Anaconda channel (`SELECT * FROM INFORMATION_SCHEMA.PACKAGES WHERE LANGUAGE = 'python'`)
2. Edit `pyproject.toml` (the `snowflake-ml-python` pin appears in the project dependencies, the `snow` group, and `ml-runtime`) and run `uv lock`
3. Commit — every compute path and the Registry pick up the new versions

To install locally for development: `uv sync --group ml-runtime`

### Versioning

Every artifact the pipeline produces is versioned, and each model version links back to the exact data and code that produced it.

| Artifact | Version scheme | Behavior |
|----------|----------------|----------|
| Feature view | `feature_store.view_version` (`V1`, `V2`, ...) | Immutable. A rerun with the same definition reuses the existing version (no rebuild). If the feature columns change and the version wasn't bumped, the run fails with a message to bump `view_version`. Bump it for any change to the feature SQL, including logic-only changes that keep the same columns. |
| Training Dataset | `<view_version>_<UTC timestamp>` (e.g. `V1_20261018_225022`) | A new point-in-time snapshot per training run, so retrains see fresh data. |
| Model | `V1`, `V2`, ... per environment | Incremented on each run that passes the gate; rejected candidates are dropped. PROD receives the same version name as the STAGE candidate it promotes. |

Each model version records the Dataset version and feature view in its lineage and comment, plus the git commit it was trained from (`git:<sha>`).

**Retention.** After each successful STAGE and PROD deploy, `source/utils/retention.py` keeps the newest `retention.keep_model_versions` model versions and `retention.keep_dataset_versions` Dataset versions per environment. It never removes the default version, a version with an inference service or a model monitor, or a Dataset referenced by a remaining version's lineage (including PROD versions that point at STAGE Datasets). Preview with `--dry-run`. Feature view versions only change when you bump `view_version`, so they aren't pruned.

## Serving

### Batch

`source/pipeline/batch_inference_dag.py --deploy --env prod` creates a scheduled Task (warehouse stored procedure) that scores `serving.batch.input_table` (default: the training feature view) with the model's **default** version and appends to `serving.batch.output_table`. Each run adds `MODEL_VERSION`, `SCORED_AT`, and `RUN_ID`, so you can always tell which version produced a score, and the monitor gets a history. Because it follows the default version, promotion and rollback need no Task changes.

### Real-time

Each PROD version gets its own inference service (`<MODEL_NAME>_SERVICE_<VERSION>`) behind one Gateway. `gateway-config.yml` holds the traffic split; edit the weights and merge to `main` to shift traffic. Whenever a split is applied, every model service that isn't a gateway target is dropped, so only versions taking traffic use compute. Any registered version can be redeployed by the rollback workflow.

To call the endpoint:

```sql
-- Ingress URL of the gateway (or of a single service)
SHOW ENDPOINTS IN SERVICE SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V3;
```

```bash
# Authenticate with a programmatic access token (PAT). Method names map to URL
# paths with underscores as dashes (predict_proba -> /predict-proba).
curl -X POST "https://<ingress-url>/predict-proba" \
  -H "Authorization: Snowflake Token=\"$PAT\"" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {
        "index": [0],
        "columns": ["TOTAL_TXN_COUNT", "AVG_TXN_AMOUNT", "MAX_TXN_AMOUNT", "STDDEV_TXN_AMOUNT",
                    "UNIQUE_MERCHANTS", "ACTIVE_DAYS", "LATE_NIGHT_TXN_RATIO", "CREDIT_SCORE",
                    "ACCOUNT_AGE_DAYS", "ANNUAL_INCOME"],
        "data": [[20, 150.5, 500.0, 100.0, 10, 30, 0.05, 700, 365, 80000]]}}'
```

By default only the service owner can call the endpoint; grant another role access with the service role `ALL_ENDPOINTS_USAGE`. Authorization failures return 404. Use a PAT: in testing, connector session tokens passed authentication but the ingress returned HTTP 500. `notebooks/06_feature_store_and_rest_inference.ipynb` walks through the full client flow: fetch features from the Feature Store, call the gateway over REST, and check parity with warehouse scoring.

The same REST checks run automatically after every PROD deploy, traffic shift, and rollback (`scripts/smoke_test_endpoint.py`), and as live pytest cases you can run against PROD:

```bash
SNOWFLAKE_DATABASE=SNOW_MLOPS_PROD SNOWFLAKE_WAREHOUSE=SNOW_MLOPS_PROD_WH uv run pytest tests/test_endpoint.py -v
```

Both issue a 1-day PAT for the connected user (restricted to its current role) and remove it afterwards, unless `SNOWFLAKE_PAT` is set.

Callers supply the feature values themselves; Snowflake also offers request-time feature lookup from an online feature store (`create_service(feature_sources_per_function=...)`, public preview), which this template does not configure. See [Deploy models for real-time inference](https://docs.snowflake.com/en/developer-guide/snowflake-ml/inference/real-time-inference-rest-api).

## Monitoring

`source/utils/setup_model_monitor.py --env prod` creates a Snowflake Model Monitor for the default version:

- **Source** — a view over `BATCH_PREDICTIONS` for that version. With `monitor.actuals` set, the view joins ground-truth labels so the monitor reports performance (AUC, precision, recall; RMSE/MAE for regression) as well as drift.
- **Baseline** — `baseline: auto` snapshots the promotion-time validation scoring as the expected distribution, so drift metrics work from day one.
- **Alert** — a Snowflake Alert logs to `DRIFT_ALERTS` when `drift_metric` exceeds `drift_threshold`, and emails `notification_recipients` when `notification_integration` is set.

## CI/CD Workflows

| Trigger | What Happens | Human Approval |
|---------|-------------|----------------|
| PR to `main` | Code quality (lint + format + tests) | Merge requires passing checks |
| Push to `main` | Code quality → train → gate (vs PROD champion) → register → validate batch/monitor → PROD deploy → REST smoke test (auto-rollback on failure) | Required for PROD |
| Weekly cron | Retrain on fresh data → register candidate → GitHub Issue | Manual promote via `promote_only` |
| Manual dispatch | Promote existing candidate or full retrain | Required for PROD |
| Merge to `main` touching `gateway-config.yml` (or manual dispatch) | Apply gateway traffic allocation across model versions | Merge review |
| Rollback dispatch | Revert PROD to previous version | Immediate |

## Extending This Template

**Add a new pipeline step:**
1. Create `source/<name>/` with a logic module exposing `run(session) -> dict` and an `entrypoint.py`
2. Add an entry to `STEPS` in `source/pipeline/train_dag.py` (module, config key), append it to `STEP_ORDER`, and add the package to `STAGED_PACKAGES`
3. Add a `pipeline_steps.<name>` block to `source/configs/config.yml`

No new dispatch code is needed — the DAG builds the warehouse or SPCS wrapper generically and chains steps in `STEP_ORDER`.

**Switch to a different model:**
1. Set `model.task_type`, `predict_function`, `id_columns`, and `label_column` in `source/configs/config.yml`
2. Replace `build_dataset()` and `train_model()` in `source/model_training/training.py`
3. Update feature columns and Feature View SQL in `source/feature_engineering/feature_views.py`
4. Set `quality_gate.metrics` and `quality_gate.champion.metric` to metrics produced for your task type (see `source/model_training/metrics.py`)
5. Point `monitor.prediction_column` and `monitor.actuals` at your model's output and labels

## Documentation

Open `docs/docs.html` in a browser for a detailed Level 300 walkthrough covering the Task DAG, ML Jobs, Feature Store, Model Registry, inference deployment, CI/CD pipeline, monitoring, and lessons learned.

## License

Apache-2.0
