"""Training pipeline Task DAG.

Pipeline steps:
  1. FEATURE_ENG  -- source/feature_engineering/feature_engineering.py
  2. TRAIN_MODEL  -- source/model_training/training.py
  3. EVALUATE     -- source/evaluate/evaluation.py

Each step exposes run(session) -> dict and independently selects its compute
via pipeline_steps.<step>.compute in source/configs/config.yml:

    warehouse -> anonymous stored procedure on a virtual warehouse
    spcs      -> ML Job (@remote) on a container compute pool

Both paths call the same run(session), so switching compute is a config change.

Usage:
    python source/pipeline/train_dag.py --deploy  --env dev
    python source/pipeline/train_dag.py --execute --env dev
    python source/pipeline/train_dag.py --status  --env dev
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

_SOURCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SOURCE_DIR))

from config import (
    ML_RUNTIME_DEPS,
    PYPI_ACCESS_INTEGRATION,
    RETRAIN_CONFIG,
    resolve_env_config,
    step_compute,
)
from snowpark_session import create_snowpark_session

# ─── Pipeline steps ──────────────────────────────────────────────────────────
# To add a step: create source/<name>/ exposing run(session), add an entry
# here, add pipeline_steps.<config_key> to config.yml, and append to STEP_ORDER.

DAG_NAME = "MLOPS_DEMO_TASK"

# Task names are a downstream contract: source/utils/wait_for_task.py polls for
# MLOPS_DEMO_TASK$FEATURE_ENG / $TRAIN_MODEL / $EVALUATE.
STEPS = {
    "FEATURE_ENG": {
        "module": "feature_engineering.feature_engineering",
        "config_key": "feature_eng",
    },
    "TRAIN_MODEL": {
        "module": "model_training.training",
        "config_key": "train_model",
    },
    "EVALUATE": {
        "module": "evaluate.evaluation",
        "config_key": "evaluate",
    },
}

STEP_ORDER = ["FEATURE_ENG", "TRAIN_MODEL", "EVALUATE"]

# Source packaged into source.zip / mounted into ML Job containers.
STAGED_PACKAGES = ["feature_engineering", "model_training", "evaluate", "serving"]
STAGED_MODULES = ["config.py", "snowpark_session.py"]

# Every step runs on the same pinned versions (pyproject.toml ml-runtime):
# pip from PyPI for ML Jobs, the Snowflake Anaconda channel for warehouse
# stored procedures.
PIP_REQUIREMENTS = ML_RUNTIME_DEPS + ["pyyaml"]
SP_PACKAGES = ["snowflake-snowpark-python", "pyyaml"] + ML_RUNTIME_DEPS

# ─── Source packaging ────────────────────────────────────────────────────────


def build_source_zip(out_dir: Path) -> Path:
    """Build the single source archive both compute paths import from.

    config.py and configs/config.yml sit side by side at the archive root so
    config.py can locate its YAML when imported from inside the zip.
    """
    zip_path = out_dir / "source.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pkg in STAGED_PACKAGES:
            for fpath in (_SOURCE_DIR / pkg).rglob("*.py"):
                if "__pycache__" not in fpath.parts:
                    zf.write(fpath, str(fpath.relative_to(_SOURCE_DIR)))

        for mod in STAGED_MODULES:
            zf.write(_SOURCE_DIR / mod, mod)

        for cfg_file in (_SOURCE_DIR / "configs").glob("*.yml"):
            zf.write(cfg_file, f"configs/{cfg_file.name}")

    return zip_path


# ─── Compute dispatch ────────────────────────────────────────────────────────
# Step modules read config.py at import time, and config.py honors these
# environment variables. Neither container inherits the deploying shell's
# environment (SPCS even sets its own SNOWFLAKE_DATABASE/SCHEMA), so each
# wrapper pins them to the target environment before importing the step.


def _git_sha() -> str:
    """Commit the DAG is deployed from (CI: GITHUB_SHA; local: git HEAD)."""
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=_SOURCE_DIR
        ).stdout.strip()
    except Exception:
        return "local"


def runtime_env(cfg: dict) -> dict:
    # Steps run in Snowflake (ML Job / stored procedure), not on the CI runner,
    # so anything they read from the environment must be forwarded here.
    return {
        "SNOWFLAKE_DATABASE": cfg["database"],
        "SNOWFLAKE_SCHEMA": cfg["schema"],
        "SNOWFLAKE_WAREHOUSE": cfg["warehouse"],
        "SNOWFLAKE_COMPUTE_POOL": cfg["compute_pool"],
        "GITHUB_SHA": _git_sha(),  # model version lineage: code -> model
        "ML_RUNTIME_DEPS": json.dumps(ML_RUNTIME_DEPS),  # log_model pins (no pyproject in the zip)
    }


def _remote_def(task_name: str, cfg: dict, job_stage: str, source_zip: Path):
    """SPCS: an ML Job running on a container compute pool."""
    from snowflake.ml.jobs import remote

    module = STEPS[task_name]["module"]
    pool = cfg["compute_pool"]
    env = runtime_env(cfg)
    pins = list(ML_RUNTIME_DEPS)

    # A local .zip passed to imports= is uploaded as-is and placed on sys.path.
    @remote(
        pool,
        stage_name=job_stage,
        pip_requirements=PIP_REQUIREMENTS,
        external_access_integrations=[PYPI_ACCESS_INTEGRATION] if PYPI_ACCESS_INTEGRATION else None,
        imports=[str(source_zip)],
        name=task_name,
    )
    def run_step() -> str:
        import importlib
        import json
        import os

        from snowflake.snowpark import Session

        os.environ.update(env)
        from config import verify_pinned_versions

        print(f"Runtime versions: {verify_pinned_versions(pins)}")
        session = Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {env['SNOWFLAKE_WAREHOUSE']}").collect()

        mod = importlib.import_module(module)
        return json.dumps(mod.run(session))

    return run_step


def sproc_definition(module: str, env: dict, imports_clause: str, packages_clause: str) -> str:
    """Anonymous stored procedure: pin env vars, verify versions, import module, call run()."""
    pins = list(ML_RUNTIME_DEPS)
    return f"""
WITH task_handler_sp AS PROCEDURE ()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION=3.12
IMPORTS=({imports_clause})
PACKAGES=({packages_clause})
HANDLER='compute'
AS $$
import importlib, json, os

def compute(session):
    os.environ.update({env!r})
    from config import verify_pinned_versions
    verify_pinned_versions({pins!r})
    step = importlib.import_module("{module}")
    return json.dumps(step.run(session))
$$
CALL task_handler_sp()"""


def _sproc_def(task_name: str, cfg: dict, imports_clause: str, packages_clause: str) -> str:
    """Warehouse: an anonymous stored procedure."""
    return sproc_definition(STEPS[task_name]["module"], runtime_env(cfg), imports_clause, packages_clause)


def _task_def(task_name: str, cfg: dict, job_stage: str, source_zip: Path, imports_clause: str, packages_clause: str):
    """Return the DAGTask definition for a step, based on its configured compute."""
    mode = step_compute(STEPS[task_name]["config_key"])

    if mode == "spcs":
        if not cfg.get("compute_pool"):
            raise ValueError(
                f"Step '{task_name}' is configured for spcs but no compute_pool is set for this environment"
            )
        print(f"    {task_name}: spcs (compute pool: {cfg['compute_pool']})")
        return _remote_def(task_name, cfg, job_stage, source_zip)

    print(f"    {task_name}: warehouse ({cfg['warehouse']})")
    return _sproc_def(task_name, cfg, imports_clause, packages_clause)


# ─── DAG deployment ──────────────────────────────────────────────────────────


def _parse_schedule():
    if RETRAIN_CONFIG.get("enabled", "false") != "true":
        return None
    schedule_str = RETRAIN_CONFIG.get("schedule", "")
    if not schedule_str:
        return None
    from snowflake.core.task import Cron

    tokens = schedule_str.replace("USING CRON ", "").strip().split()
    tz = tokens[-1]
    cron_expr = " ".join(tokens[:-1])
    print(f"  Schedule: {cron_expr} ({tz})")
    return Cron(cron_expr, tz)


def deploy_dag(env: str):
    """Deploy the ML training pipeline DAG."""
    from snowflake.core import Root
    from snowflake.core._common import CreateMode
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask

    cfg = resolve_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Deploying DAG: {db}.{schema}.{DAG_NAME}")

    session.sql(f"CREATE STAGE IF NOT EXISTS {db}.{schema}.PIPELINE_STAGE").collect()
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.PIPELINE_RESULTS (
            STEP VARCHAR, STATUS VARCHAR, RESULT VARIANT, CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()

    pipeline_stage = f"@{db}.{schema}.PIPELINE_STAGE"

    # ML Job payloads are written under <stage>/<TASK_NAME>/app/ and are not
    # cleaned up between deploys. Stale files from earlier package layouts can
    # shadow current code, so clear them before rebuilding.
    for name in STEP_ORDER:
        session.sql(f"REMOVE {pipeline_stage}/{name}/app/").collect()

    dag_schedule = _parse_schedule()
    packages_clause = ",".join(f"'{p}'" for p in SP_PACKAGES)

    # The zip must exist until DAGOperation.deploy() returns: @remote uploads
    # its imports when the definition is built, the warehouse path reads it
    # from the stage at run time.
    with tempfile.TemporaryDirectory() as tmp:
        source_zip = build_source_zip(Path(tmp))
        print("  Uploading source.zip to stage...")
        session.file.put(str(source_zip), f"{pipeline_stage}/", auto_compress=False, overwrite=True)
        imports_clause = f"'{pipeline_stage}/source.zip'"

        print("  Step compute:")
        with DAG(
            DAG_NAME,
            warehouse=wh,
            schedule=dag_schedule,
            stage_location=pipeline_stage,
            use_func_return_value=True,
        ) as dag:
            tasks = {
                name: DAGTask(
                    name,
                    definition=_task_def(name, cfg, pipeline_stage, source_zip, imports_clause, packages_clause),
                    warehouse=wh,
                )
                for name in STEP_ORDER
            }
            for upstream, downstream in zip(STEP_ORDER, STEP_ORDER[1:]):
                tasks[upstream] >> tasks[downstream]

        root = Root(session)
        schema_ref = root.databases[db].schemas[schema]
        DAGOperation(schema_ref).deploy(dag, mode=CreateMode.or_replace)
    print(f"  DAG deployed: {DAG_NAME}")

    if os.getenv("RESUME_SCHEDULE", "false").lower() == "true" and dag_schedule:
        session.sql(f"ALTER TASK {db}.{schema}.{DAG_NAME} RESUME").collect()
        print("  Root task resumed (scheduled retraining active)")

    session.close()
    print("Deploy complete.")


def execute_dag(env: str):
    from snowflake.core import Root

    cfg = resolve_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {cfg['warehouse']}").collect()

    Root(session).databases[db].schemas[schema].tasks[DAG_NAME].execute()
    print(f"Executed: {db}.{schema}.{DAG_NAME}")
    session.close()


def show_status(env: str):
    cfg = resolve_env_config(env)
    db = cfg["database"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {cfg['warehouse']}").collect()

    print(f"Task history for {db} (last 24h):\n")
    rows = session.sql(f"""
        SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME, RETURN_VALUE, ERROR_MESSAGE
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
            SCHEDULED_TIME_RANGE_START => DATEADD('hour', -24, CURRENT_TIMESTAMP()),
            RESULT_LIMIT => 20
        ))
        WHERE DATABASE_NAME = '{db}'
        ORDER BY SCHEDULED_TIME DESC
    """).collect()

    if not rows:
        print("  No task runs found in the last 24 hours.")
    else:
        for row in rows:
            ts = str(row["SCHEDULED_TIME"])[:19]
            ret = row["RETURN_VALUE"] or ""
            err = row["ERROR_MESSAGE"] or ""
            print(f"  [{ts}] {row['NAME']}: {row['STATE']} {ret[:80]} {err[:80]}")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Task DAG (Python SDK)")
    parser.add_argument("--deploy", action="store_true", help="Deploy the Task DAG")
    parser.add_argument("--execute", action="store_true", help="Execute the Task DAG")
    parser.add_argument("--status", action="store_true", help="Show recent task history")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "dev"), choices=["dev", "stage", "prod"])
    args = parser.parse_args()

    if args.deploy:
        deploy_dag(args.env)
    elif args.execute:
        execute_dag(args.env)
    elif args.status:
        show_status(args.env)
    else:
        parser.print_help()
