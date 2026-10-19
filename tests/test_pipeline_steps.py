"""Tests for the pipeline step contract and per-step compute configuration."""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import (
    PIPELINE_STEPS,
    VALID_COMPUTE_MODES,
    resolve_env_config,
    step_compute,
)
from pipeline.train_dag import (
    DAG_NAME,
    STAGED_MODULES,
    STAGED_PACKAGES,
    STEP_ORDER,
    STEPS,
)

# Step packages in the source/ directory
STEP_PACKAGES = ["feature_engineering", "model_training", "evaluate"]

# Full dotted module paths (must match STEPS registry)
STEP_MODULE_PATHS = {
    "FEATURE_ENG": "feature_engineering.feature_engineering",
    "TRAIN_MODEL": "model_training.training",
    "EVALUATE": "evaluate.evaluation",
}


# ─── Step contract ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task_name", STEP_MODULE_PATHS)
def test_step_module_exposes_run(task_name):
    import importlib

    mod = importlib.import_module(STEP_MODULE_PATHS[task_name])
    assert hasattr(mod, "run"), f"{STEP_MODULE_PATHS[task_name]} has no run()"
    assert callable(mod.run)


@pytest.mark.parametrize("task_name", STEP_MODULE_PATHS)
def test_step_run_signature(task_name):
    """run() must accept exactly (session) — config via module-level imports."""
    import importlib

    mod = importlib.import_module(STEP_MODULE_PATHS[task_name])
    params = list(inspect.signature(mod.run).parameters)
    assert params == ["session"], f"{STEP_MODULE_PATHS[task_name]}.run has signature {params}, expected ['session']"


@pytest.mark.parametrize("pkg", STEP_PACKAGES)
def test_step_has_entrypoint(pkg):
    entrypoint = Path(__file__).resolve().parent.parent / "source" / pkg / "entrypoint.py"
    assert entrypoint.exists(), f"Missing entrypoint: {entrypoint}"


# ─── Per-step compute configuration ───────────────────────────────────────────


def test_all_steps_have_compute_configured():
    for spec in STEPS.values():
        assert spec["config_key"] in PIPELINE_STEPS, f"pipeline_steps is missing '{spec['config_key']}' in config.yml"


@pytest.mark.parametrize("task_name", STEPS)
def test_compute_modes_are_valid(task_name):
    assert step_compute(STEPS[task_name]["config_key"]) in VALID_COMPUTE_MODES


def test_step_compute_rejects_invalid_mode(monkeypatch):
    import config

    monkeypatch.setitem(config.PIPELINE_STEPS, "feature_eng", {"compute": "kubernetes"})
    with pytest.raises(ValueError, match="invalid compute"):
        config.step_compute("feature_eng")


def test_step_compute_defaults_to_warehouse():
    assert step_compute("step_that_does_not_exist") == "warehouse"


# ─── Environment resolution ───────────────────────────────────────────────────


@pytest.mark.parametrize("env", ["dev", "stage", "prod"])
def test_resolve_env_config_has_infra_keys(env):
    cfg = resolve_env_config(env)
    for key in ("env", "database", "schema", "warehouse", "compute_pool", "source_database", "source_schema"):
        assert key in cfg, f"resolve_env_config('{env}') missing key: {key}"


def test_resolve_env_config_rejects_unknown_env():
    with pytest.raises(ValueError, match="Unknown environment"):
        resolve_env_config("qa")


def test_environments_target_distinct_databases(monkeypatch):
    # SNOWFLAKE_* env vars intentionally override per-env names (CI relies on
    # this), so clear them to test the YAML defaults.
    for var in ("SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_COMPUTE_POOL"):
        monkeypatch.delenv(var, raising=False)
    databases = {env: resolve_env_config(env)["database"] for env in ("dev", "stage", "prod")}
    assert len(set(databases.values())) == 3, f"Environments share a database: {databases}"


def test_source_data_is_always_prod():
    for env in ("dev", "stage", "prod"):
        assert resolve_env_config(env)["source_database"] == "SNOW_MLOPS_PROD"


# ─── DAG wiring ──────────────────────────────────────────────────────────────


def test_dag_step_registry_matches_module_paths():
    for task_name, spec in STEPS.items():
        assert task_name in STEP_MODULE_PATHS, f"Unknown task in registry: {task_name}"
        assert spec["module"] == STEP_MODULE_PATHS[task_name]


def test_dag_task_names_unchanged():
    """source/utils/wait_for_task.py polls for these exact task names."""
    assert DAG_NAME == "MLOPS_DEMO_TASK"
    assert set(STEPS) == {"FEATURE_ENG", "TRAIN_MODEL", "EVALUATE"}


def test_step_order_covers_all_steps():
    assert set(STEP_ORDER) == set(STEPS)
    assert len(STEP_ORDER) == len(STEPS)
    assert STEP_ORDER[0] == "FEATURE_ENG"


def test_staged_packages_exist():
    source_dir = Path(__file__).resolve().parent.parent / "source"
    for pkg in STAGED_PACKAGES:
        assert (source_dir / pkg).is_dir(), f"Staged package not found: {pkg}"
        assert (source_dir / pkg / "__init__.py").exists(), f"'{pkg}' has no __init__.py"


def test_staged_modules_exist():
    source_dir = Path(__file__).resolve().parent.parent / "source"
    for mod in STAGED_MODULES:
        assert (source_dir / mod).exists(), f"Staged module not found: {mod}"


def test_step_packages_are_staged():
    for pkg in STEP_PACKAGES:
        assert pkg in STAGED_PACKAGES, f"Step '{pkg}' is not staged"


def test_all_compute_paths_use_the_same_pins():
    from config import ML_RUNTIME_DEPS, MODEL_DEPS
    from pipeline.train_dag import PIP_REQUIREMENTS, SP_PACKAGES, runtime_env

    assert any(d.startswith("snowflake-ml-python==") for d in ML_RUNTIME_DEPS), "snowflake-ml-python must be pinned"
    for dep in ML_RUNTIME_DEPS:
        assert "==" in dep, f"ml-runtime dependency must be an exact pin: {dep}"
        assert dep in PIP_REQUIREMENTS, f"ML Jobs missing pin: {dep}"
        assert dep in SP_PACKAGES, f"stored procedures missing pin: {dep}"
    assert all(d in ML_RUNTIME_DEPS for d in MODEL_DEPS)
    assert not any(d.startswith("snowflake-ml-python") for d in MODEL_DEPS)
    env = runtime_env({"database": "D", "schema": "S", "warehouse": "W", "compute_pool": "P"})
    assert json.loads(env["ML_RUNTIME_DEPS"]) == ML_RUNTIME_DEPS


def test_client_pin_matches_runtime_pin():
    """CI/local client and Snowflake runtimes must run the same snowflake-ml-python."""
    import tomllib

    from config import ML_RUNTIME_DEPS

    pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    runtime_pin = next(d for d in ML_RUNTIME_DEPS if d.startswith("snowflake-ml-python"))
    assert runtime_pin in pyproject["project"]["dependencies"]
    assert runtime_pin in pyproject["dependency-groups"]["snow"]


def test_verify_pinned_versions():
    from importlib.metadata import version

    from config import verify_pinned_versions

    have = version("pyyaml")
    assert verify_pinned_versions([f"pyyaml=={have}", "unpinned-pkg"]) == {"pyyaml": have}
    with pytest.raises(RuntimeError, match="pinned"):
        verify_pinned_versions(["pyyaml==0.0.1"])


def test_env_resolution_defined_in_one_place():
    source_dir = Path(__file__).resolve().parent.parent / "source"
    monitor_src = (source_dir / "monitoring" / "model_monitor.py").read_text()
    assert "resolve_env_config" in monitor_src
    assert "SNOW_MLOPS_STAGE_WH" not in monitor_src


@pytest.mark.parametrize("env", ["dev", "stage", "prod"])
def test_warehouse_step_pins_target_environment(env):
    """The stored procedure must set the target env before config.py imports.

    Otherwise a STAGE deploy runs against the YAML defaults (DEV).
    """
    from pipeline.train_dag import _sproc_def

    cfg = resolve_env_config(env)
    sql = _sproc_def("TRAIN_MODEL", cfg, "'@x/source.zip'", "'pyyaml'")
    assert cfg["database"] in sql
    assert sql.index("os.environ.update") < sql.index("importlib.import_module")
