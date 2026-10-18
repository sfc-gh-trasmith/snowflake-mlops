"""Model Monitor: ML Observability for the deployed model.

Uses Snowflake's native MODEL MONITOR over the appended batch predictions
(serving.batch.output_table).

  Source    A view over the predictions for the monitored version. When
            monitor.actuals is configured, the view joins ground-truth labels,
            enabling performance metrics (AUC, precision, RMSE, ...) as well
            as drift.
  Baseline  monitor.baseline: auto snapshots the promotion-time validation
            scoring (<output>_VALIDATION) as the expected distribution, so
            drift metrics (PSI, Jensen-Shannon, Wasserstein, Difference of
            Means) work out of the box.
  Alert     A Snowflake Alert logs to DRIFT_ALERTS when drift exceeds the
            threshold, and optionally emails via a notification integration.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ID_COLUMNS,
    MODEL_NAME,
    MONITOR_CONFIG,
    PREDICT_FUNCTION,
    SOURCE_DATABASE,
    SOURCE_SCHEMA,
    TASK_TYPE,
    batch_output_table,
    resolve_env_config,
)

VALID_DRIFT_METRICS = frozenset({"POPULATION_STABILITY_INDEX", "JENSEN_SHANNON", "WASSERSTEIN", "DIFFERENCE_OF_MEANS"})
ACTUAL_COLUMN = "ACTUAL_LABEL"


def get_env_config(env: str) -> dict:
    """Environment database/warehouse for the monitor (source/configs/config.yml)."""
    try:
        return resolve_env_config(env)
    except ValueError:
        return resolve_env_config("dev")


def _target(env: str) -> tuple[str, str, str]:
    cfg = get_env_config(env)
    return cfg["database"], cfg["schema"], cfg["warehouse"]


def actuals_config() -> dict | None:
    actuals = MONITOR_CONFIG.get("actuals") or {}
    return actuals if actuals.get("table") else None


def labeled_select_sql(predictions_table: str, version: str | None) -> str:
    """SELECT over predictions (optionally one version), joined to actuals if configured."""
    where = f" WHERE p.MODEL_VERSION = '{version}'" if version else ""
    actuals = actuals_config()
    if not actuals:
        return f"SELECT p.* FROM {predictions_table} p{where}"

    keys = list(actuals.get("join_keys") or ID_COLUMNS)
    table = actuals["table"]
    table_fqn = table if "." in table else f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.{table}"
    key_list = ", ".join(keys)
    join = " AND ".join(f"p.{k} = a.{k}" for k in keys)
    return (
        f"SELECT p.*, a.{ACTUAL_COLUMN} FROM {predictions_table} p "
        f"LEFT JOIN (SELECT {key_list}, {actuals['label_sql']} AS {ACTUAL_COLUMN} "
        f"FROM {table_fqn} GROUP BY {key_list}) a ON {join}{where}"
    )


def monitor_columns_sql() -> str:
    """Column clauses for CREATE MODEL MONITOR, by task type and actuals."""
    pred_col = MONITOR_CONFIG["prediction_column"]
    clauses = [f"PREDICTION_SCORE_COLUMNS = ('{pred_col}')"]
    if ID_COLUMNS:
        # Predictions are appended per run, so the entity key alone repeats;
        # RUN_ID makes each row unique.
        clauses.append("ID_COLUMNS = (" + ", ".join(f"'{c}'" for c in [*ID_COLUMNS, "RUN_ID"]) + ")")
    if actuals_config():
        if TASK_TYPE == "regression":
            clauses.append(f"ACTUAL_SCORE_COLUMNS = ('{ACTUAL_COLUMN}')")
        elif TASK_TYPE == "binary_classification":
            clauses.append(f"ACTUAL_CLASS_COLUMNS = ('{ACTUAL_COLUMN}')")
        # multiclass: Model Monitor performance metrics need a single score
        # column, so only drift is monitored.
    return "\n            ".join(clauses)


def _table_exists(session, fqn: str) -> bool:
    db, schema, name = fqn.split(".", 2)
    rows = session.sql(f"SHOW TABLES LIKE '{name}' IN SCHEMA {db}.{schema}").collect()
    return bool(rows)


def _resolve_version(session, db: str, schema: str) -> str:
    version = os.getenv("MODEL_VERSION")
    if version:
        return version
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {db}.{schema}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {db}.{schema}")
    return models[0]["default_version_name"]


def prepare_sources(session, db: str, schema: str, version: str) -> dict:
    """Create the monitor source view and (optionally) the baseline table."""
    predictions = batch_output_table(db, schema)
    validation = predictions + "_VALIDATION"

    # The scheduled scoring task may not have run yet on a fresh deploy.
    if not _table_exists(session, predictions):
        if not _table_exists(session, validation):
            raise RuntimeError(f"Neither {predictions} nor {validation} exists — run batch inference first")
        session.sql(f"CREATE TABLE {predictions} LIKE {validation}").collect()
        print(f"  Created empty {predictions} (scheduled scoring will append)")

    source_view = f"{predictions}_MONITOR_SOURCE"
    session.sql(f"CREATE OR REPLACE VIEW {source_view} AS {labeled_select_sql(predictions, version)}").collect()

    baseline_cfg = MONITOR_CONFIG.get("baseline")
    baseline = None
    if baseline_cfg == "auto":
        if _table_exists(session, validation):
            baseline = f"{predictions}_BASELINE"
            session.sql(f"CREATE OR REPLACE TABLE {baseline} AS {labeled_select_sql(validation, version)}").collect()
        else:
            print(f"  Baseline auto: {validation} not found — drift metrics disabled")
    elif baseline_cfg:
        baseline = baseline_cfg if "." in baseline_cfg else f"{db}.{schema}.{baseline_cfg}"

    return {"source": source_view, "baseline": baseline}


def create_monitor(session, env: str = "prod") -> str:
    """Create (or recreate for a new version) the model monitor. Returns its name."""
    db, schema, wh = _target(env)
    monitor_name = MONITOR_CONFIG["monitor_name"]
    version = _resolve_version(session, db, schema)
    sources = prepare_sources(session, db, schema, version)

    existing = session.sql(f"SHOW MODEL MONITORS LIKE '{monitor_name}' IN SCHEMA {db}.{schema}").collect()
    if existing:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        current = _get_model_version_from_desc(desc[0]) if desc else ""
        if current == version:
            print(f"  Monitor already tracks {MODEL_NAME}/{version}.")
            _ensure_active(session, db, schema, monitor_name)
            return monitor_name
        print(f"  Recreating monitor for {version} (was {current or 'unknown'})...")
        session.sql(f"DROP MODEL MONITOR IF EXISTS {db}.{schema}.{monitor_name}").collect()

    baseline_clause = f"\n            BASELINE = {sources['baseline']}" if sources["baseline"] else ""
    print(f"  Creating monitor {monitor_name}: {MODEL_NAME}/{version}")
    print(f"    Source: {sources['source']}")
    print(f"    Actuals: {'yes' if actuals_config() else 'no (drift only)'}")
    print(f"    Baseline: {sources['baseline'] or 'none (drift metrics disabled)'}")

    session.sql(f"""
        CREATE MODEL MONITOR {db}.{schema}.{monitor_name} WITH
            MODEL = {db}.{schema}.{MODEL_NAME}
            VERSION = '{version}'
            FUNCTION = '{PREDICT_FUNCTION}'
            SOURCE = {sources["source"]}
            WAREHOUSE = {wh}
            REFRESH_INTERVAL = '{MONITOR_CONFIG["refresh_interval"]}'
            AGGREGATION_WINDOW = '{MONITOR_CONFIG["aggregation_window"]}'
            TIMESTAMP_COLUMN = {MONITOR_CONFIG["timestamp_column"]}
            {monitor_columns_sql()}{baseline_clause}
    """).collect()
    _ensure_active(session, db, schema, monitor_name)
    return monitor_name


def _get_model_version_from_desc(row) -> str:
    import json

    try:
        return json.loads(row.model).get("version_name", "")
    except Exception:
        return ""


def _ensure_active(session, db: str, schema: str, monitor_name: str):
    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        state = (desc[0].monitor_state if desc else None) or "UNKNOWN"
        if state == "SUSPENDED":
            print("  Resuming suspended monitor...")
            session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} RESUME").collect()
        else:
            print(f"  Monitor state: {state}")
    except Exception as e:
        print(f"  Warning: could not check monitor state: {e}")


def _has_baseline(session, db: str, schema: str, monitor_name: str) -> bool:
    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
        baseline = (desc[0].baseline or "") if desc else ""
        return bool(baseline) and "NOT_SET" not in baseline.upper()
    except Exception:
        return False


def alert_action_sql(db: str, schema: str, metric: str, threshold: float) -> str:
    """Alert THEN clause: log to DRIFT_ALERTS, plus email when configured."""
    insert = (
        f"INSERT INTO {db}.{schema}.DRIFT_ALERTS (DRIFT_METRIC, THRESHOLD, STATUS) "
        f"VALUES ('{metric}', {threshold}, 'THRESHOLD_EXCEEDED')"
    )
    integration = MONITOR_CONFIG.get("notification_integration")
    recipients = MONITOR_CONFIG.get("notification_recipients") or []
    if not integration or not recipients:
        return insert
    subject = f"[{db}] {MODEL_NAME} drift alert: {metric} > {threshold}"
    body = f"Model monitor {MONITOR_CONFIG['monitor_name']} in {db}.{schema} exceeded the drift threshold."
    email = f"CALL SYSTEM$SEND_EMAIL('{integration}', '{','.join(recipients)}', '{subject}', '{body}')"
    return f"EXECUTE IMMEDIATE $$ BEGIN {insert}; {email}; END; $$"


def create_drift_alert(session, env: str = "prod") -> str | None:
    """Create a Snowflake Alert for drift breaches. None if the monitor has no baseline."""
    db, schema, wh = _target(env)
    monitor_name = MONITOR_CONFIG["monitor_name"]
    if not _has_baseline(session, db, schema, monitor_name):
        print("  Drift alert: skipped (monitor has no baseline)")
        return None

    alert_name = MONITOR_CONFIG.get("alert_name", f"{MODEL_NAME}_DRIFT_ALERT")
    metric = MONITOR_CONFIG.get("drift_metric", "POPULATION_STABILITY_INDEX")
    threshold = float(MONITOR_CONFIG.get("drift_threshold", 0.25))
    column = MONITOR_CONFIG["prediction_column"]
    if metric not in VALID_DRIFT_METRICS:
        raise ValueError(f"Invalid drift_metric '{metric}'. Must be one of: {sorted(VALID_DRIFT_METRICS)}")

    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.DRIFT_ALERTS (
            ALERT_TS TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            DRIFT_METRIC VARCHAR, THRESHOLD FLOAT, STATUS VARCHAR
        )
    """).collect()
    session.sql(f"""
        CREATE OR REPLACE ALERT {db}.{schema}.{alert_name}
            WAREHOUSE = {wh}
            SCHEDULE = '{MONITOR_CONFIG.get("alert_schedule", "1440 MINUTE")}'
            IF (EXISTS (
                SELECT 1 FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
                    '{db}.{schema}.{monitor_name}', '{metric}', '{column}', '1 DAY'
                )) WHERE METRIC_VALUE > {threshold}
                  AND EVENT_TIMESTAMP > DATEADD('DAY', -1, CURRENT_TIMESTAMP())
            ))
            THEN {alert_action_sql(db, schema, metric, threshold)}
    """).collect()
    session.sql(f"ALTER ALERT {db}.{schema}.{alert_name} RESUME").collect()
    print(f"  Alert {alert_name}: {metric} > {threshold} (every {MONITOR_CONFIG.get('alert_schedule')})")
    return alert_name


def drop_drift_alert(session, env: str = "prod"):
    db, schema, _ = _target(env)
    alert_name = MONITOR_CONFIG.get("alert_name", f"{MODEL_NAME}_DRIFT_ALERT")
    session.sql(f"DROP ALERT IF EXISTS {db}.{schema}.{alert_name}").collect()
    print(f"  Alert {alert_name} dropped.")


def query_drift(session, env: str = "prod", lookback_days: int = 30) -> list:
    """Daily drift metric for the last N days (empty if no baseline)."""
    db, schema, _ = _target(env)
    monitor_name = MONITOR_CONFIG["monitor_name"]
    if not _has_baseline(session, db, schema, monitor_name):
        return []
    return session.sql(f"""
        SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
            '{db}.{schema}.{monitor_name}', '{MONITOR_CONFIG.get("drift_metric", "POPULATION_STABILITY_INDEX")}',
            '{MONITOR_CONFIG["prediction_column"]}', '1 DAY',
            DATEADD('DAY', -{int(lookback_days)}, CURRENT_TIMESTAMP()), CURRENT_TIMESTAMP()
        ))
    """).collect()


def get_monitor_status(session, env: str = "prod") -> dict:
    db, schema, _ = _target(env)
    try:
        desc = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{MONITOR_CONFIG['monitor_name']}").collect()
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
    db, schema, _ = _target(env)
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{MONITOR_CONFIG['monitor_name']} SUSPEND").collect()


def resume_monitor(session, env: str = "prod"):
    db, schema, _ = _target(env)
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{MONITOR_CONFIG['monitor_name']} RESUME").collect()
