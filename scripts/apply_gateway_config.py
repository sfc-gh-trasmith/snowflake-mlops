"""Apply gateway traffic configuration from gateway-config.yml.

Reads the YAML file, validates all services exist and are READY,
validates weights sum to 100, applies ALTER GATEWAY, and drops every
per-version inference service that is not a gateway target. A service
receiving no traffic only costs compute, and any version can be redeployed
from the registry (rollback workflow).

Called by .github/workflows/traffic-shift.yml on push to main.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

import yaml
from config import MODEL_NAME, REALTIME_CONFIG, SCHEMA, env_database, env_setting
from snowpark_session import create_snowpark_session

CONFIG_PATH = Path(__file__).resolve().parent.parent / "gateway-config.yml"
GATEWAY_NAME = REALTIME_CONFIG.get("gateway_name", f"{MODEL_NAME}_GATEWAY")
PROD_DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("prod"))
PROD_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)


def service_name(version: str) -> str:
    """Naming convention for per-version inference services."""
    return f"{MODEL_NAME}_SERVICE_{version.strip().upper()}"


def load_config(path=None) -> dict:
    """Load gateway-config.yml ({"targets": [{"service", "weight"}, ...]})."""
    p = Path(path) if path else CONFIG_PATH
    config = yaml.safe_load(p.read_text()) or {}
    config["targets"] = config.get("targets") or []
    return config


def validate_weights(targets: list[dict]) -> list[str]:
    """Return problems with a target list (empty = valid)."""
    problems = []
    if not targets:
        problems.append("no targets")
    total = sum(int(t.get("weight", 0)) for t in targets)
    if total != 100:
        problems.append(f"weights sum to {total}, must be 100")
    if any(int(t.get("weight", 0)) <= 0 for t in targets):
        problems.append("weights must be > 0 (remove a target to stop serving it)")
    names = [t.get("service") for t in targets]
    if len(set(names)) != len(names):
        problems.append("duplicate services")
    return problems


def get_current_gateway_targets(session, db, schema, gateway_name) -> list[dict]:
    """Current gateway split as [{"service", "weight"}] (empty if no gateway)."""
    try:
        rows = session.sql(f"DESC GATEWAY {db}.{schema}.{gateway_name}").collect()
    except Exception:
        return []
    targets = []
    for line in (rows[0]["spec"] if rows else "").split("\n"):
        line = line.strip()
        if line.startswith("value:"):
            fqn = line.split("value:")[1].strip().split("!")[0].strip()
            targets.append({"service": fqn.split(".")[-1], "weight": None})
        elif line.startswith("weight:") and targets:
            targets[-1]["weight"] = int(line.split("weight:")[1])
    return targets


def get_current_gateway_services(session, db, schema, gateway_name) -> list[str]:
    """Services currently behind the gateway."""
    return [t["service"] for t in get_current_gateway_targets(session, db, schema, gateway_name)]


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


def gateway_exists(session, db, schema, gateway_name) -> bool:
    rows = session.sql(f"SHOW GATEWAYS LIKE '{gateway_name}' IN SCHEMA {db}.{schema}").collect()
    return len(rows) > 0


def apply_gateway(session, db, schema, gateway_name, targets):
    """Create the gateway on first deploy, otherwise ALTER it to the declared split."""
    spec_targets = ""
    for t in targets:
        fqn = f"{db}.{schema}.{t['service']}!inference"
        spec_targets += f"""
              - type: endpoint
                value: {fqn}
                weight: {t["weight"]}"""

    verb = "ALTER" if gateway_exists(session, db, schema, gateway_name) else "CREATE"
    print(f"  {verb} GATEWAY {db}.{schema}.{gateway_name}")
    session.sql(f"""
        {verb} GATEWAY {db}.{schema}.{gateway_name}
        FROM SPECIFICATION $$
          spec:
            type: traffic_split
            split_type: custom
            targets:{spec_targets}
        $$
    """).collect()


def model_services(session, db, schema) -> list[str]:
    """All per-version inference services for this model (<MODEL>_SERVICE_<V>)."""
    rows = session.sql(f"SHOW SERVICES LIKE '{MODEL_NAME}_SERVICE_%' IN SCHEMA {db}.{schema}").collect()
    return [r["name"] for r in rows]


def unused_services(existing: list[str], targets: list[dict]) -> list[str]:
    """Services that exist but receive no gateway traffic."""
    serving = {t["service"] for t in targets}
    return [s for s in existing if s not in serving]


def cleanup_unused_services(session, db, schema, targets) -> list[str]:
    """Drop every model service the gateway doesn't route traffic to.

    Covers services removed from the config, services left behind by older
    deploys, and canaries that never made it into the gateway.
    """
    dropped = []
    for svc in unused_services(model_services(session, db, schema), targets):
        print(f"  Dropping unused service: {svc}")
        session.sql(f"DROP SERVICE IF EXISTS {db}.{schema}.{svc}").collect()
        dropped.append(svc)
    if not dropped:
        print("  No unused services.")
    return dropped


def parse_targets_arg(targets_str):
    """Parse 'V6:80,V7:20' into target dicts with full service names."""
    targets = []
    for pair in targets_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid target '{pair}' — expected 'VERSION:WEIGHT' (e.g., 'V7:100')")
        version, weight = pair.split(":", 1)
        targets.append({"service": service_name(version), "weight": int(weight.strip())})
    return targets


def update_config_file(targets, path=None):
    """Write targets back to gateway-config.yml, preserving the header comment."""
    p = Path(path) if path else CONFIG_PATH
    header = [line for line in (p.read_text().splitlines() if p.exists() else []) if line.startswith("#")]
    lines = header + ["", "targets:"]
    for t in targets:
        lines += [f"  - service: {t['service']}", f"    weight: {t['weight']}"]
    p.write_text("\n".join(lines) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply gateway traffic configuration")
    parser.add_argument(
        "--targets",
        help='Traffic targets as "VERSION:WEIGHT" pairs (e.g., "V6:80,V7:20" or "V7:100")',
    )
    args = parser.parse_args()

    gateway_name, db, schema = GATEWAY_NAME, PROD_DATABASE, PROD_SCHEMA

    if args.targets:
        targets = parse_targets_arg(args.targets)
        update_config_file(targets)
        print(f"  Updated gateway-config.yml from --targets: {args.targets}")
    else:
        targets = load_config()["targets"]
        if not targets:
            # Fresh template: nothing is serving yet. deploy-prod writes the
            # first target after the initial real-time deployment.
            print("gateway-config.yml has no targets yet; nothing to apply.")
            return

    problems = validate_weights(targets)
    if problems:
        print(f"ERROR: invalid gateway-config.yml: {'; '.join(problems)}")
        sys.exit(1)

    print("=" * 60)
    print("GATEWAY TRAFFIC SHIFT")
    print("=" * 60)
    print(f"  Gateway: {gateway_name}")
    for t in targets:
        print(f"    {t['service']}: {t['weight']}%")

    session = create_snowpark_session()
    wh = os.getenv("SNOWFLAKE_WAREHOUSE", env_setting("prod", "warehouse"))
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

    # Drop services that no longer receive traffic
    print("\n[4/4] Dropping services with no gateway traffic...")
    dropped = cleanup_unused_services(session, db, schema, targets)

    # Write to Job Summary if in CI
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n## Gateway Traffic Shift\n\n")
            f.write("| Service | Weight |\n|---------|--------|\n")
            for t in targets:
                f.write(f"| `{t['service']}` | {t['weight']}% |\n")
            if dropped:
                f.write("\nDropped (no traffic): " + ", ".join(f"`{s}`" for s in dropped) + "\n")

    print("\n" + "=" * 60)
    print("TRAFFIC SHIFT COMPLETE")
    for t in targets:
        print(f"  {t['service']}: {t['weight']}%")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
