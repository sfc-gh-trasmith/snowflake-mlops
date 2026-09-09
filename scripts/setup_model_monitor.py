"""Set up Model Monitor — CI entry point.

Creates or updates the model monitor in the target environment.
Called by deploy-prod after batch inference has populated BATCH_PREDICTIONS.

Optionally creates a Snowflake Alert for drift threshold breaches when
a baseline table is configured.

Usage:
    uv run python scripts/setup_model_monitor.py [--env prod]
    uv run python scripts/setup_model_monitor.py --env stage --validate-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import MONITOR_CONFIG
from monitoring.model_monitor import (
    create_drift_alert,
    create_monitor,
    drop_drift_alert,
    get_env_config,
    get_monitor_status,
)
from snowpark_session import create_snowpark_session


def main():
    parser = argparse.ArgumentParser(description="Set up model monitor")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "prod"), choices=["dev", "stage", "prod"])
    parser.add_argument("--validate-only", action="store_true", help="Create, verify, then drop (for STAGE validation)")
    parser.add_argument("--skip-alert", action="store_true", help="Skip drift alert creation")
    args = parser.parse_args()

    print("=" * 60)
    if args.validate_only:
        print("MODEL MONITOR VALIDATION (create → verify → drop)")
    else:
        print("MODEL MONITOR SETUP")
    print("=" * 60)

    session = create_snowpark_session()

    # Create or update monitor
    print(f"\n[1/3] Creating/updating monitor (env={args.env})...")
    monitor_name = create_monitor(session, env=args.env)

    # Verify status
    print("\n[2/3] Verifying monitor status...")
    status = get_monitor_status(session, env=args.env)
    print(f"  Monitor: {monitor_name}")
    print(f"  State: {status.get('state', 'UNKNOWN')}")
    print(f"  Model version: {status.get('model_version', 'N/A')}")
    print(f"  Refresh interval: {status.get('refresh_interval', 'N/A')}")
    print(f"  Baseline: {status.get('baseline', 'NOT_SET')}")

    # Drift alert
    alert_name = None
    if not args.skip_alert:
        print("\n[3/3] Drift alert setup...")
        alert_name = create_drift_alert(session, env=args.env)
        if alert_name:
            print(f"  Drift metric: {MONITOR_CONFIG.get('drift_metric', 'PSI')}")
            print(f"  Threshold: {MONITOR_CONFIG.get('drift_threshold', 0.25)}")
        else:
            print("  Drift alert: skipped (no baseline)")
    else:
        print("\n[3/3] Drift alert: skipped (--skip-alert)")

    # Validate-only: drop after verification (STAGE validation)
    if args.validate_only:
        cfg = get_env_config(args.env)
        db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
        schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
        if alert_name:
            drop_drift_alert(session, env=args.env)
        session.sql(f"DROP MODEL MONITOR IF EXISTS {db}.{schema}.{monitor_name}").collect()
        print("\n  Monitor dropped (validate-only mode).")
        print("\n" + "=" * 60)
        print("MODEL MONITOR VALIDATION PASSED")
        print("=" * 60)
        session.close()
        return

    # Write to Job Summary
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n## Model Monitor\n\n")
            f.write("| Property | Value |\n|----------|-------|\n")
            f.write(f"| Monitor | `{monitor_name}` |\n")
            f.write(f"| State | {status.get('state', 'UNKNOWN')} |\n")
            f.write(f"| Model version | {status.get('model_version', 'N/A')} |\n")
            f.write(f"| Refresh interval | {status.get('refresh_interval', 'N/A')} |\n")
            f.write(f"| Baseline | {status.get('baseline', 'NOT_SET')} |\n")
            f.write(f"| Drift metric | {MONITOR_CONFIG.get('drift_metric', 'N/A')} |\n")
            f.write(f"| Drift threshold | {MONITOR_CONFIG.get('drift_threshold', 'N/A')} |\n")
            f.write(f"| Alert | {'`' + alert_name + '`' if alert_name else 'disabled (no baseline)'} |\n")

    print("\n" + "=" * 60)
    print("MODEL MONITOR ACTIVE")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
