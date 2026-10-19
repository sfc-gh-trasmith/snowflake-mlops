"""Capture and restore PROD serving state for automatic rollback in deploy-prod.

  --capture   records the current default model version and gateway split as
              step outputs (GITHUB_OUTPUT), before promotion changes anything.
  --restore   puts both back after a failed deploy: default version first
              (batch scoring and the monitor follow it), then the gateway split,
              which also drops the services the restored split doesn't use.

Usage:
  python scripts/prod_state.py --capture
  python scripts/prod_state.py --restore --version V4 --targets "V4:100"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_gateway_config import GATEWAY_NAME, get_current_gateway_targets
from config import MODEL_NAME, SCHEMA, env_database
from snowpark_session import create_snowpark_session

PROD_DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("prod"))
PROD_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)
MODEL_FQN = f"{PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}"


def targets_arg(targets: list[dict]) -> str:
    """[{"service": "<MODEL>_SERVICE_V4", "weight": 100}] -> "V4:100"."""
    return ",".join(f"{t['service'].rsplit('_SERVICE_', 1)[-1]}:{t['weight']}" for t in targets)


def capture(session) -> dict:
    rows = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    state = {
        "default_version": rows[0]["default_version_name"] if rows else "",
        "gateway_targets": targets_arg(get_current_gateway_targets(session, PROD_DATABASE, PROD_SCHEMA, GATEWAY_NAME)),
    }
    print(
        f"PROD before deploy: default={state['default_version'] or 'none'} gateway={state['gateway_targets'] or 'none'}"
    )
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in state.items())
    return state


def restore(session, version: str, targets: str):
    if not version:
        # First deployment: nothing to go back to. The failed version stays
        # registered but nothing new is routed to it beyond what failed.
        print("No previous PROD version (first deployment); nothing to restore.")
        return
    session.sql(f"ALTER MODEL {MODEL_FQN} SET DEFAULT_VERSION = {version}").collect()
    print(f"Default version restored: {version}")
    if targets:
        import subprocess

        # Same code path as the traffic-shift and rollback workflows.
        script = Path(__file__).resolve().parent / "apply_gateway_config.py"
        subprocess.run([sys.executable, str(script), "--targets", targets], check=True)
        print(f"Gateway restored: {targets}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--restore", action="store_true")
    parser.add_argument("--version", default="", help="Default version to restore")
    parser.add_argument("--targets", default="", help='Gateway split to restore, e.g. "V4:100"')
    args = parser.parse_args()

    session = create_snowpark_session()
    if args.capture:
        capture(session)
    else:
        restore(session, args.version.strip(), args.targets.strip())
        summary = os.getenv("GITHUB_STEP_SUMMARY")
        if summary and args.version:
            with open(summary, "a") as f:
                f.write(
                    f"\n## Automatic rollback\n\nDeploy failed after promotion. PROD restored to "
                    f"`{args.version}`" + (f", gateway `{args.targets}`" if args.targets else "") + ".\n"
                )
    session.close()


if __name__ == "__main__":
    main()
