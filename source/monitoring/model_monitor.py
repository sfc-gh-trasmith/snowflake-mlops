"""Model Monitor: create and manage ML Observability for the fraud detector.

Uses Snowflake's native ModelMonitor (CREATE MODEL MONITOR) to track
prediction drift and feature distribution shifts over time.

The monitor reads from the BATCH_PREDICTIONS table (scored by batch inference)
and checks for distribution shifts at a configurable interval.

When a baseline table is configured, drift metrics (PSI, Jensen-Shannon,
Wasserstein, Difference of Means) are computed against the baseline. A
Snowflake Alert fires when the chosen metric exceeds the configured threshold.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, MONITOR_CONFIG

VALID_DRIFT_METRICS = frozenset({"POPULATION_STABILITY_INDEX", "JENSEN_SHANNON", "WASSERSTEIN", "DIFFERENCE_OF_MEANS"})


def get_env_config(env: str) -> dict:
    """Get environment-specific database/warehouse for monitor."""
    envs = {
        "dev": {"database": "SNOW_MLOPS_DEV", "warehouse": "SNOW_MLOPS_DEV_WH"},
        "stage": {"database": "SNOW_MLOPS_STAGE", "warehouse": "SNOW_MLOPS_STAGE_WH"},
        "prod": {"database": "SNOW_MLOPS_PROD", "warehouse": "SNOW_MLOPS_PROD_WH"},
    }
    return envs.get(env, envs["dev"])


def create_monitor(session, env: str = "prod") -> str:
    """Create the model monitor if it doesn't exist. Returns the monitor name."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    wh = cfg["warehouse"]

    monitor_name = MONITOR_CONFIG["monitor_name"]
    source_table = f"{db}.{schema}.{MONITOR_CONFIG['source_table']}"
    ts_col = MONITOR_CONFIG["timestamp_column"]
    pred_cols = MONITOR_CONFIG["prediction_columns"]
    refresh = MONITOR_CONFIG["refresh_interval"]
    agg_window = MONITOR_CONFIG["aggregation_window"]
    function_name = MONITOR_CONFIG["function_name"]
    baseline_table = MONITOR_CONFIG.get("baseline_table")

    # Get current default model version
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {db}.{schema}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {db}.{schema}")
    version = models[0]["default_version_name"]

    pred_cols_sql = ", ".join(f"'{c}'" for c in pred_cols)

    # Check if monitor already exists
    existing = session.sql(f"SHOW MODEL MONITORS LIKE '{monitor_name}' IN SCHEMA {db}.{schema}").collect()

    if existing:
        print(f"  Monitor {monitor_name} already exists. Verifying state...")
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        current_version = _get_model_version_from_desc(desc[0]) if desc else ""
        if current_version != version:
            print(f"  Updating monitor to track version {version} (was {current_version})...")
            session.sql(f"DROP MODEL MONITOR IF EXISTS {db}.{schema}.{monitor_name}").collect()
        else:
            print(f"  Monitor is tracking {MODEL_NAME}/{version}. No changes needed.")
            _ensure_active(session, db, schema, monitor_name)
            _set_baseline_if_configured(session, db, schema, monitor_name, baseline_table)
            return monitor_name

    # Create the monitor
    print(f"  Creating monitor: {monitor_name}")
    print(f"    Model: {MODEL_NAME}/{version}")
    print(f"    Source: {source_table}")
    print(f"    Refresh: {refresh}, Aggregation: {agg_window}")

    baseline_clause = ""
    if baseline_table:
        baseline_fqn = f"{db}.{schema}.{baseline_table}" if "." not in baseline_table else baseline_table
        baseline_clause = f"\n            BASELINE = {baseline_fqn}"

    session.sql(f"""
        CREATE MODEL MONITOR {db}.{schema}.{monitor_name} WITH
            MODEL = {db}.{schema}.{MODEL_NAME}
            VERSION = '{version}'
            FUNCTION = '{function_name}'
            SOURCE = {source_table}
            WAREHOUSE = {wh}
            REFRESH_INTERVAL = '{refresh}'
            AGGREGATION_WINDOW = '{agg_window}'
            TIMESTAMP_COLUMN = {ts_col}
            PREDICTION_SCORE_COLUMNS = ({pred_cols_sql}){baseline_clause}
    """).collect()

    print("  Monitor created successfully.")
    _ensure_active(session, db, schema, monitor_name)

    if baseline_table:
        print(f"  Baseline: {baseline_table}")
    else:
        print("  Baseline: not configured (drift detection disabled)")
        print("  Set MONITOR_CONFIG['baseline_table'] to enable drift metrics.")

    return monitor_name


def _get_model_version_from_desc(row) -> str:
    """Extract model version from DESC MODEL MONITOR row."""
    import json

    try:
        model_json = json.loads(row.model)
        return model_json.get("version_name", "")
    except Exception:
        return ""


def _set_baseline_if_configured(session, db: str, schema: str, monitor_name: str, baseline_table: str | None):
    """Set the baseline on an existing monitor if configured and not already set."""
    if not baseline_table:
        return
    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        if desc:
            current_baseline = desc[0].baseline or ""
            if current_baseline and "NOT_SET" not in current_baseline:
                return
        baseline_fqn = f"{db}.{schema}.{baseline_table}" if "." not in baseline_table else baseline_table
        print(f"  Setting baseline: {baseline_fqn}")
        session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} SET BASELINE = '{baseline_fqn}'").collect()
    except Exception as e:
        print(f"  Warning: could not set baseline: {e}")


def _ensure_active(session, db: str, schema: str, monitor_name: str):
    """Ensure the monitor is in ACTIVE state."""
    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        if desc:
            state = desc[0].monitor_state or "UNKNOWN"
            if state == "SUSPENDED":
                print("  Resuming suspended monitor...")
                session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} RESUME").collect()
            elif state == "ACTIVE":
                print("  Monitor state: ACTIVE")
            else:
                print(f"  Monitor state: {state}")
    except Exception as e:
        print(f"  Warning: could not check monitor state: {e}")


def create_drift_alert(session, env: str = "prod") -> str | None:
    """Create a Snowflake Alert that fires when drift exceeds the threshold.

    Returns the alert name, or None if baseline is not configured.
    """
    baseline_table = MONITOR_CONFIG.get("baseline_table")
    if not baseline_table:
        print("  Drift alert: skipped (no baseline configured)")
        return None

    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    wh = cfg["warehouse"]

    monitor_name = MONITOR_CONFIG["monitor_name"]
    alert_name = MONITOR_CONFIG.get("alert_name", "FRAUD_DETECTOR_DRIFT_ALERT")
    alert_schedule = MONITOR_CONFIG.get("alert_schedule", "1440 MINUTE")
    drift_metric = MONITOR_CONFIG.get("drift_metric", "POPULATION_STABILITY_INDEX")
    drift_column = MONITOR_CONFIG.get("drift_column", "output_feature_1")
    drift_threshold = MONITOR_CONFIG.get("drift_threshold", 0.25)

    if drift_metric not in VALID_DRIFT_METRICS:
        raise ValueError(f"Invalid drift_metric '{drift_metric}'. Must be one of: {VALID_DRIFT_METRICS}")

    # Create a log table for drift alerts
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.DRIFT_ALERTS (
            ALERT_TS TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            DRIFT_METRIC VARCHAR,
            THRESHOLD FLOAT,
            STATUS VARCHAR
        )
    """).collect()

    # Create the alert
    monitor_fqn = f"'{db}.{schema}.{monitor_name}'"
    print(f"  Creating drift alert: {alert_name}")
    print(f"    Metric: {drift_metric} > {drift_threshold}")
    print(f"    Schedule: every {alert_schedule}")

    session.sql(f"""
        CREATE OR REPLACE ALERT {db}.{schema}.{alert_name}
            WAREHOUSE = {wh}
            SCHEDULE = '{alert_schedule}'
            IF (EXISTS (
                SELECT 1 FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
                    {monitor_fqn}, '{drift_metric}', '{drift_column}', '1 DAY'
                )) WHERE METRIC_VALUE > {drift_threshold}
                  AND EVENT_TIMESTAMP > DATEADD('DAY', -1, CURRENT_TIMESTAMP())
            ))
            THEN
                INSERT INTO {db}.{schema}.DRIFT_ALERTS (DRIFT_METRIC, THRESHOLD, STATUS)
                VALUES ('{drift_metric}', {drift_threshold}, 'THRESHOLD_EXCEEDED')
    """).collect()

    session.sql(f"ALTER ALERT {db}.{schema}.{alert_name} RESUME").collect()
    print(f"  Alert {alert_name} created and resumed.")
    return alert_name


def drop_drift_alert(session, env: str = "prod"):
    """Drop the drift alert."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    alert_name = MONITOR_CONFIG.get("alert_name", "FRAUD_DETECTOR_DRIFT_ALERT")
    session.sql(f"DROP ALERT IF EXISTS {db}.{schema}.{alert_name}").collect()
    print(f"  Alert {alert_name} dropped.")


def query_drift(session, env: str = "prod", lookback_days: int = 30) -> list:
    """Query drift metrics for the last N days. Returns empty list if no baseline."""
    baseline_table = MONITOR_CONFIG.get("baseline_table")
    if not baseline_table:
        print("  No baseline configured — drift metrics unavailable.")
        return []

    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")

    monitor_name = MONITOR_CONFIG["monitor_name"]
    drift_metric = MONITOR_CONFIG.get("drift_metric", "POPULATION_STABILITY_INDEX")
    drift_column = MONITOR_CONFIG.get("drift_column", "output_feature_1")

    monitor_fqn = f"'{db}.{schema}.{monitor_name}'"
    rows = session.sql(f"""
        SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
            {monitor_fqn}, '{drift_metric}', '{drift_column}', '1 DAY',
            DATEADD('DAY', -{lookback_days}, CURRENT_TIMESTAMP()), CURRENT_TIMESTAMP()
        ))
    """).collect()
    return rows


def get_monitor_status(session, env: str = "prod") -> dict:
    """Get the current monitor status and latest metrics."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]

    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        if not desc:
            return {"exists": False}
        row = desc[0]
        return {
            "exists": True,
            "state": row.monitor_state or "UNKNOWN",
            "model_version": _get_model_version_from_desc(row),
            "refresh_interval": row.refresh_interval or "",
            "baseline": row.baseline or "NOT_SET",
        }
    except Exception:
        return {"exists": False}


def suspend_monitor(session, env: str = "prod"):
    """Suspend the monitor (e.g., during maintenance)."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} SUSPEND").collect()
    print(f"  Monitor {monitor_name} suspended.")


def resume_monitor(session, env: str = "prod"):
    """Resume a suspended monitor."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} RESUME").collect()
    print(f"  Monitor {monitor_name} resumed.")
