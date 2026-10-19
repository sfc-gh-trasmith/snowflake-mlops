"""Centralized configuration for the Snowflake MLOps template.

Values are loaded from source/configs/config.yml and re-exported as Python
constants so every module has one import surface.

Environment variables override infrastructure names, which is how CI targets
DEV / STAGE / PROD with the same code.
"""

import json
import os
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _load_yml(name: str) -> dict:
    """Read a YAML file from configs/, whether on disk or inside a zip.

    Inside a Snowflake stored procedure or ML Job, this module is imported from
    a staged source.zip, so plain open() cannot reach configs/. The module's
    loader can read from either location.
    """
    path = _CONFIGS_DIR / name
    try:
        # Use __file__'s directory as-is: the zip loader matches on the exact
        # archive path it imported from, so a resolve()d path would not match.
        raw = __loader__.get_data(os.path.join(os.path.dirname(__file__), "configs", name))
    except OSError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e
    return yaml.safe_load(raw) or {}


def _str_bools(cfg: dict) -> dict:
    """Coerce YAML booleans to lowercase strings.

    Several consumers compare against the string "true" (e.g.
    RETRAIN_CONFIG["enabled"] == "true"), so booleans are normalized to
    preserve that contract.
    """
    return {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in cfg.items()}


_cfg = _load_yml("config.yml")

# ─── Target environment (where the pipeline WRITES) ──────────────────────────
DATABASE = os.getenv("SNOWFLAKE_DATABASE", _cfg["database"])
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", _cfg["schema"])
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", _cfg["warehouse"])
COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", _cfg["compute_pool"])

# ─── Source environment (where raw data LIVES -- always PROD) ────────────────
SOURCE_DATABASE = os.getenv("SNOWFLAKE_SOURCE_DATABASE", _cfg["source_database"])
SOURCE_SCHEMA = os.getenv("SNOWFLAKE_SOURCE_SCHEMA", _cfg["source_schema"])

# ─── Model ───────────────────────────────────────────────────────────────────
VALID_TASK_TYPES = ("binary_classification", "multiclass", "regression")

MODEL_CONFIG = dict(_cfg["model"])
MODEL_NAME = MODEL_CONFIG["name"]
TASK_TYPE = MODEL_CONFIG["task_type"]
PREDICT_FUNCTION = MODEL_CONFIG["predict_function"]
ID_COLUMNS = list(MODEL_CONFIG.get("id_columns") or [])
LABEL_COLUMN = MODEL_CONFIG["label_column"]
if TASK_TYPE not in VALID_TASK_TYPES:
    raise ValueError(f"model.task_type '{TASK_TYPE}' must be one of {VALID_TASK_TYPES}")

# ─── Feature Store ───────────────────────────────────────────────────────────
_fs_cfg = _cfg["feature_store"]
FEATURE_VIEW_NAME = _fs_cfg["view_name"]
FEATURE_VIEW_VERSION = _fs_cfg["view_version"]

# ─── Quality gate ─────────────────────────────────────────────────────────────────────
# {"metrics": {name: {"min"|"max": v}}, "champion": {...}}
GATE_CONFIG = _cfg["quality_gate"]
GATE_RULES = dict(GATE_CONFIG.get("metrics") or {})
CHAMPION_CONFIG = dict(GATE_CONFIG.get("champion") or {"enabled": False})

# ─── Serving ──────────────────────────────────────────────────────────────────────────
SERVING_CONFIG = _cfg.get("serving", {})
BATCH_CONFIG = dict(SERVING_CONFIG.get("batch") or {"enabled": False})
REALTIME_CONFIG = dict(SERVING_CONFIG.get("realtime") or {"enabled": False})


def serving_enabled(mode: str) -> bool:
    """True if a serving mode ("batch" or "realtime") is enabled in config."""
    if mode not in ("batch", "realtime"):
        raise ValueError(f"Unknown serving mode: {mode}")
    return bool((SERVING_CONFIG.get(mode) or {}).get("enabled", False))


def batch_input_table(database: str, schema: str) -> str:
    """Fully-qualified batch scoring input (defaults to the feature view table)."""
    table = BATCH_CONFIG.get("input_table")
    if table:
        return table if "." in table else f"{database}.{schema}.{table}"
    return f'{database}.{schema}."{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"'


def batch_output_table(database: str, schema: str) -> str:
    return f"{database}.{schema}.{BATCH_CONFIG.get('output_table', 'BATCH_PREDICTIONS')}"


# ─── Data checks ───────────────────────────────────────────────────────────────────────
DATA_CHECKS = dict(_cfg.get("data_checks") or {})

# ─── Training hyperparameters (read by the training step) ────────────────────
TRAINING_PARAMS = dict(_cfg["training_params"])

# ─── ML runtime dependencies ─────────────────────────────────────────────────
# Read from pyproject.toml [dependency-groups] ml-runtime -- the single source
# of truth for train/serve version alignment. Deliberately NOT in YAML: the
# same pins feed both the training job and log_model(conda_dependencies=).


def _read_ml_runtime_deps() -> list[str]:
    # pyproject.toml is only present in the repo checkout (local dev and CI).
    # Pipeline steps run from a staged source.zip without it, so the DAG
    # forwards the pins in ML_RUNTIME_DEPS (see train_dag.runtime_env).
    if os.getenv("ML_RUNTIME_DEPS"):
        return json.loads(os.environ["ML_RUNTIME_DEPS"])
    pyproject = _ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return []
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["dependency-groups"]["ml-runtime"]


ML_RUNTIME_DEPS = _read_ml_runtime_deps()

# Registry models get the same pins minus snowflake-ml-python, which log_model
# adds itself at the version of the client that logged the model.
MODEL_DEPS = [d for d in ML_RUNTIME_DEPS if not d.startswith("snowflake-ml-python")]

# External access integration that lets ML Jobs pip-install the pins above.
# Without it, pip cannot reach PyPI and the job silently keeps the versions
# preinstalled in the runtime image -- verify_pinned_versions() catches that.
PYPI_ACCESS_INTEGRATION = _cfg.get("pypi_access_integration")


def verify_pinned_versions(pins: list[str]) -> dict:
    """Fail if installed package versions differ from the '==' pins.

    Called at the start of every pipeline step (ML Job or stored procedure) so
    a runtime that didn't honor the pins fails loudly instead of training or
    serving on a different library version than CI and the Registry expect.
    """
    from importlib.metadata import version

    installed, problems = {}, []
    for spec in pins:
        if "==" not in spec:
            continue
        name, want = (s.strip() for s in spec.split("==", 1))
        installed[name] = have = version(name)
        if have != want:
            problems.append(f"{name} {have} installed, {want} pinned")
    if problems:
        raise RuntimeError(
            "Runtime does not match pyproject.toml ml-runtime pins: "
            + "; ".join(problems)
            + ". For ML Jobs, check that pypi_access_integration is set and granted to this role."
        )
    return installed


# ─── Per-step compute selection ──────────────────────────────────────────────
# {"feature_eng": {"compute": "spcs"}, "train_model": {...}, "evaluate": {...}}
PIPELINE_STEPS = _cfg.get("pipeline_steps", {})

VALID_COMPUTE_MODES = ("warehouse", "spcs")


def step_compute(step: str) -> str:
    """Return the configured compute mode for a pipeline step."""
    mode = PIPELINE_STEPS.get(step, {}).get("compute", "warehouse")
    if mode not in VALID_COMPUTE_MODES:
        raise ValueError(f"Step '{step}' has invalid compute '{mode}'. Use one of {VALID_COMPUTE_MODES}.")
    return mode


# ─── Environment resolution ──────────────────────────────────────────────────
_ENVIRONMENTS = _cfg.get("environments", {})


def env_setting(env: str, key: str) -> str:
    """Configured infrastructure name for an environment, ignoring env-var overrides.

    Use this to reach a *different* environment than the one this process
    targets (e.g. the PROD champion from a STAGE run), where the pinned
    SNOWFLAKE_* variables would otherwise point back at the current one.
    """
    if env not in _ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env}. Use one of {sorted(_ENVIRONMENTS)}.")
    return _ENVIRONMENTS[env][key]


def env_database(env: str) -> str:
    return env_setting(env, "database")


def resolve_env_config(env: str) -> dict:
    """Resolve the full config dict for an environment (dev / stage / prod).

    Returns a flat dict of primitives, safe to serialize as JSON and inject
    into a remote container or stored procedure.
    """
    if env not in _ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env}. Use one of {sorted(_ENVIRONMENTS)}.")

    env_cfg = _ENVIRONMENTS[env]
    return {
        "env": env,
        "database": os.getenv("SNOWFLAKE_DATABASE", env_cfg["database"]),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", _cfg["schema"]),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", env_cfg["warehouse"]),
        "compute_pool": os.getenv("SNOWFLAKE_COMPUTE_POOL", env_cfg["compute_pool"]),
        "source_database": SOURCE_DATABASE,
        "source_schema": SOURCE_SCHEMA,
        "model_name": MODEL_NAME,
        "feature_view_name": FEATURE_VIEW_NAME,
        "feature_view_version": FEATURE_VIEW_VERSION,
        "customer_features_refresh": _fs_cfg.get("customer_features_refresh", "1 hour"),
        "transaction_features_refresh": _fs_cfg.get("transaction_features_refresh", "1 hour"),
        "training_params": TRAINING_PARAMS,
        "experiment": _str_bools(_cfg.get("experiment", {})),
    }


# ─── Pipeline configuration ──────────────────────────────────────────────────
PIPELINE_CONFIG = {
    "database": DATABASE,
    "schema": SCHEMA,
    "source_database": SOURCE_DATABASE,
    "source_schema": SOURCE_SCHEMA,
    "warehouse": WAREHOUSE,
    "compute_pool": COMPUTE_POOL,
    "model_name": MODEL_NAME,
    "feature_view_name": FEATURE_VIEW_NAME,
    "feature_view_version": FEATURE_VIEW_VERSION,
}

# ─── Model Monitor configuration ─────────────────────────────────────────────
MONITOR_CONFIG = _str_bools(_cfg["monitor"])

# ─── Experiment Tracking configuration ───────────────────────────────────────
EXPERIMENT_CONFIG = _str_bools(_cfg["experiment"])

# ─── Scheduled Retraining configuration ──────────────────────────────────────
RETRAIN_CONFIG = _str_bools(_cfg["retrain"])

# ─── Retention (model and Dataset versions) ──────────────────────────────────
RETENTION_CONFIG = _str_bools(_cfg.get("retention") or {"enabled": False})
