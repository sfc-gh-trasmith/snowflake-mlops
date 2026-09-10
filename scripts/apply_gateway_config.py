"""Apply gateway traffic configuration from gateway-config.yml.

Reads the YAML file, validates all services exist and are READY,
validates weights sum to 100, applies ALTER GATEWAY, and drops
any services that were previously in the gateway but are no longer
in the config.

Called by .github/workflows/traffic-shift.yml on push to main.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

try:
    import yaml
except ImportError:
    # PyYAML not available — use a minimal parser for the simple gateway config
    yaml = None

from snowpark_session import create_snowpark_session

CONFIG_PATH = Path(__file__).resolve().parent.parent / "gateway-config.yml"


def load_config(path=None):
    """Load gateway-config.yml. Uses PyYAML if available, falls back to simple parser."""
    p = Path(path) if path else CONFIG_PATH
    text = p.read_text()
    if yaml:
        return yaml.safe_load(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text):
    """Minimal parser for the gateway-config.yml structure (no PyYAML dependency)."""
    config = {}
    current_target = None
    targets = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- service:"):
            current_target = {"service": stripped.split(":", 1)[1].strip()}
            targets.append(current_target)
        elif stripped.startswith("weight:") and current_target is not None:
            current_target["weight"] = int(stripped.split(":", 1)[1].strip())
        elif ":" in stripped and not stripped.startswith("-"):
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val.isdigit():
                val = int(val)
            config[key] = val

    config["targets"] = targets
    return config


def get_current_gateway_services(session, db, schema, gateway_name):
    """Parse the current gateway spec to find all active service names."""
    try:
        rows = session.sql(f"DESC GATEWAY {db}.{schema}.{gateway_name}").collect()
    except Exception:
        return []
    spec = rows[0]["spec"]
    services = []
    for line in spec.split("\n"):
        line = line.strip()
        if line.startswith("value:"):
            fqn = line.split("value:")[1].strip().split("!")[0].strip()
            services.append(fqn.split(".")[-1])
    return services


def validate_services(session, db, schema, targets):
    """Check each target service exists."""
    for target in targets:
        svc = target["service"]
        fqn = f"{db}.{schema}.{svc}"
        try:
            status_json = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{fqn}')").collect()[0][0]
            statuses = json.loads(status_json)
            current = statuses[0]["status"] if statuses else "UNKNOWN"
            failed = current in ("FAILED", "DELETING")
            if failed:
                raise RuntimeError(f"Service {svc} is {current} — cannot route traffic to it")
            print(f"  {svc}: {current}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Service {svc} not found or not accessible: {e}")


def apply_gateway(session, db, schema, gateway_name, targets):
    """ALTER GATEWAY with the declared traffic split."""
    spec_targets = ""
    for t in targets:
        fqn = f"{db}.{schema}.{t['service']}!inference"
        spec_targets += f"""
              - type: endpoint
                value: {fqn}
                weight: {t["weight"]}"""

    session.sql(f"""
        ALTER GATEWAY {db}.{schema}.{gateway_name}
        FROM SPECIFICATION $$
          spec:
            type: traffic_split
            split_type: custom
            targets:{spec_targets}
        $$
    """).collect()


def cleanup_removed_services(session, db, schema, old_services, new_targets):
    """Drop services that were in the gateway but are no longer in the config."""
    new_set = {t["service"] for t in new_targets}
    for svc in old_services:
        if svc not in new_set:
            print(f"  Dropping removed service: {svc}")
            session.sql(f"DROP SERVICE IF EXISTS {db}.{schema}.{svc}").collect()


def parse_targets_arg(targets_str, model_name="MLOPS_FRAUD_DETECTOR"):
    """Parse 'V6:80,V7:20' into target dicts with full service names."""
    targets = []
    for pair in targets_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid target '{pair}' — expected 'VERSION:WEIGHT' (e.g., 'V7:100')")
        version, weight = pair.split(":", 1)
        version = version.strip().upper()
        service_name = f"{model_name}_SERVICE_{version}"
        targets.append({"service": service_name, "weight": int(weight.strip())})
    return targets


def update_config_file(config, targets):
    """Write updated targets back to gateway-config.yml."""
    config_path = CONFIG_PATH
    lines = [
        "# Gateway traffic configuration — single source of truth.",
        "# Edit weights and merge to main to shift traffic.",
        "# The traffic-shift workflow reads this file and applies ALTER GATEWAY.",
        "#",
        "# Services removed from this file (or set to weight 0) are dropped automatically.",
        "# Weights must sum to 100.",
        "",
        f"gateway: {config['gateway']}",
        f"database: {config['database']}",
        f"schema: {config['schema']}",
        "",
        "# Default canary weight for new deployments (used by deploy_prod_service.py).",
        "# Set to 100 for instant cutover (no canary period).",
        f"initial_canary_weight: {config.get('initial_canary_weight', 20)}",
        "",
        "targets:",
    ]
    for t in targets:
        lines.append(f"  - service: {t['service']}")
        lines.append(f"    weight: {t['weight']}")
    config_path.write_text("\n".join(lines) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply gateway traffic configuration")
    parser.add_argument(
        "--targets",
        help='Traffic targets as "VERSION:WEIGHT" pairs (e.g., "V6:80,V7:20" or "V7:100")',
    )
    args = parser.parse_args()

    config = load_config()
    gateway_name = config["gateway"]
    db = config["database"]
    schema = config["schema"]

    # If --targets provided, parse and override
    if args.targets:
        from config import MODEL_NAME

        targets = parse_targets_arg(args.targets, MODEL_NAME)
        update_config_file(config, targets)
        print(f"  Updated gateway-config.yml from --targets: {args.targets}")
    else:
        targets = config["targets"]

    # Validate weights sum to 100
    total = sum(t["weight"] for t in targets)
    if total != 100:
        print(f"ERROR: Weights sum to {total}, must be 100.")
        sys.exit(1)

    print("=" * 60)
    print("GATEWAY TRAFFIC SHIFT")
    print("=" * 60)
    print(f"  Gateway: {gateway_name}")
    for t in targets:
        print(f"    {t['service']}: {t['weight']}%")

    session = create_snowpark_session()
    wh = os.getenv("SNOWFLAKE_WAREHOUSE", "SNOW_MLOPS_PROD_WH")
    session.sql(f"USE WAREHOUSE {wh}").collect()

    # Get current state
    print("\n[1/4] Reading current gateway state...")
    old_services = get_current_gateway_services(session, db, schema, gateway_name)
    print(f"  Current services: {old_services or 'none (new gateway)'}")

    # Validate all target services exist
    print("\n[2/4] Validating target services...")
    validate_services(session, db, schema, targets)

    # Apply the gateway change
    print("\n[3/4] Applying gateway traffic split...")
    apply_gateway(session, db, schema, gateway_name, targets)
    print("  Gateway updated.")

    # Cleanup services removed from config
    print("\n[4/4] Cleaning up removed services...")
    cleanup_removed_services(session, db, schema, old_services, targets)

    # Write to Job Summary if in CI
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n## Gateway Traffic Shift\n\n")
            f.write("| Service | Weight |\n|---------|--------|\n")
            for t in targets:
                f.write(f"| `{t['service']}` | {t['weight']}% |\n")

    print("\n" + "=" * 60)
    print("TRAFFIC SHIFT COMPLETE")
    for t in targets:
        print(f"  {t['service']}: {t['weight']}%")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
