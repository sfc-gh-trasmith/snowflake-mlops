"""PROD real-time deployment: create, validate, and set a canary traffic split.

Called by the deploy-prod workflow when serving.realtime.enabled is true.

Flow:
  1. Resolve the PROD default model version
  2. Create a versioned inference service (<MODEL>_SERVICE_<VERSION>)
  3. Wait for READY
  4. Health check on real rows from the batch input table:
       - outputs are valid for the task type
       - parity: service outputs match warehouse mv.run() for the same version
       - agreement with the incumbent service (reported, not gated)
  5. Write a canary split to gateway-config.yml (applied by apply_gateway_config.py)

All model-specific inputs come from the model signature and config, so no
feature names or model types are hardcoded here.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_gateway_config import (
    GATEWAY_NAME,
    get_current_gateway_services,
    load_config,
    service_name,
    update_config_file,
)
from config import (
    MODEL_NAME,
    PREDICT_FUNCTION,
    REALTIME_CONFIG,
    SCHEMA,
    TASK_TYPE,
    batch_input_table,
    env_database,
    env_setting,
)
from snowpark_session import create_snowpark_session

PROD_DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("prod"))
PROD_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)
PROD_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", env_setting("prod", "warehouse"))
PROD_COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", env_setting("prod", "compute_pool"))

READY_TIMEOUT_SECONDS = 600
READY_POLL_INTERVAL = 15
PARITY_TOLERANCE = 1e-4


def current_gateway_targets(session) -> list[str]:
    """Services currently behind the gateway (empty if the gateway doesn't exist)."""
    return get_current_gateway_services(session, PROD_DATABASE, PROD_SCHEMA, GATEWAY_NAME)


def wait_for_service_ready(session, name, timeout=READY_TIMEOUT_SECONDS) -> bool:
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{name}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            statuses = json.loads(session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{fqn}')").collect()[0][0])
            if any(s["status"] == "READY" for s in statuses):
                return True
            print(f"  Status: {statuses[0]['status']} (elapsed: {int(time.time() - start)}s)")
        except Exception as e:
            print(f"  Waiting for service to appear... ({e})")
        time.sleep(READY_POLL_INTERVAL)
    return False


def health_check(session, mv, new_service: str, incumbent_service: str | None) -> dict:
    """Validate the new service on real rows. Raises on invalid output or parity failure."""
    import numpy as np
    from serving.signature import input_features, run_outputs, validate_outputs

    n_rows = int(REALTIME_CONFIG.get("health_check_rows", 20))
    feats = input_features(mv, PREDICT_FUNCTION)
    sample = session.table(batch_input_table(PROD_DATABASE, PROD_SCHEMA)).select(*feats).limit(n_rows).to_pandas()
    if sample.empty:
        raise RuntimeError("Health check: batch input table returned no rows")

    service_out = run_outputs(mv, sample, PREDICT_FUNCTION, service_name=new_service)
    problems = validate_outputs(TASK_TYPE, service_out)
    if problems:
        raise RuntimeError(f"Health check: invalid outputs from {new_service}: {problems}")

    warehouse_out = run_outputs(mv, sample, PREDICT_FUNCTION)
    parity = float(np.abs(service_out - warehouse_out).max())
    if parity > PARITY_TOLERANCE:
        raise RuntimeError(f"Health check: service vs warehouse outputs differ by {parity:.6f}")

    report = {"rows": len(sample), "max_parity_diff": parity, "incumbent_agreement": None}
    if incumbent_service and incumbent_service != new_service:
        try:
            from snowflake.ml.registry import Registry

            inc_version = incumbent_service.rsplit("_SERVICE_", 1)[-1]
            inc_mv = (
                Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
                .get_model(MODEL_NAME)
                .version(inc_version)
            )
            inc_out = run_outputs(inc_mv, sample, PREDICT_FUNCTION, service_name=incumbent_service)
            if TASK_TYPE == "regression":
                report["incumbent_mean_abs_diff"] = float(np.abs(service_out - inc_out).mean())
            else:
                pick = (lambda o: o.argmax(axis=1)) if service_out.shape[1] > 1 else (lambda o: o[:, 0])
                report["incumbent_agreement"] = float((pick(service_out) == pick(inc_out)).mean())
        except Exception as e:
            print(f"  Incumbent comparison skipped: {e}")
    return report


def canary_targets(new_service: str, incumbents: list[str], weight: int) -> list[dict]:
    """New service at `weight`%, remainder to the first incumbent (or 100% if none)."""
    old = next((s for s in incumbents if s != new_service), None)
    if not old or weight >= 100:
        return [{"service": new_service, "weight": 100}]
    return [{"service": old, "weight": 100 - weight}, {"service": new_service, "weight": weight}]


def main():
    print("=" * 60)
    print("PROD REAL-TIME DEPLOYMENT: canary via gateway")
    print("=" * 60)

    from snowflake.ml.registry import Registry

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()

    print("\n[1/5] Resolving PROD default version...")
    mv = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA).get_model(MODEL_NAME).default
    version = mv.version_name
    new_service = service_name(version)
    incumbents = current_gateway_targets(session)
    print(f"  Model: {MODEL_NAME}/{version}")
    print(f"  New service: {new_service} | gateway targets: {incumbents or 'none'}")

    if incumbents == [new_service]:
        print(f"\n  {new_service} already serves 100% of traffic. Nothing to deploy.")
        session.close()
        return

    print(f"\n[2/5] Creating service {new_service}...")
    created = False
    try:
        mv.create_service(
            service_name=new_service,
            service_compute_pool=PROD_COMPUTE_POOL,
            image_build_compute_pool=PROD_COMPUTE_POOL,
            ingress_enabled=True,
            min_instances=int(REALTIME_CONFIG.get("min_instances", 1)),
            max_instances=int(REALTIME_CONFIG.get("max_instances", 2)),
            autocapture=True,
        )
        created = True
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        print("  Service already exists — verifying readiness.")

    try:
        print("\n[3/5] Waiting for READY...")
        if not wait_for_service_ready(session, new_service):
            raise RuntimeError(f"Service {new_service} did not become READY within {READY_TIMEOUT_SECONDS}s")

        print("\n[4/5] Health check...")
        incumbent = next((s for s in incumbents if s != new_service), None)
        report = health_check(session, mv, new_service, incumbent)
        print(f"  PASSED: {report}")
    except Exception:
        # A service that never takes traffic only costs compute. Leave services
        # the gateway already routes to alone.
        if created and new_service not in incumbents:
            print(f"  Dropping failed service {new_service}")
            session.sql(f"DROP SERVICE IF EXISTS {PROD_DATABASE}.{PROD_SCHEMA}.{new_service}").collect()
        raise

    print("\n[5/5] Writing canary split to gateway-config.yml...")
    weight = int(REALTIME_CONFIG.get("canary_weight", 20))
    targets = canary_targets(new_service, incumbents or [t["service"] for t in load_config()["targets"]], weight)
    update_config_file(targets)
    for t in targets:
        print(f"    {t['service']}: {t['weight']}%")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"\n## Real-time Canary: `{new_service}`\n\n| Check | Value |\n|-------|-------|\n")
            f.write(f"| Rows checked | {report['rows']} |\n")
            f.write(f"| Service vs warehouse max diff | {report['max_parity_diff']:.2e} |\n")
            if report.get("incumbent_agreement") is not None:
                f.write(f"| Agreement with incumbent | {report['incumbent_agreement']:.1%} |\n")
            if report.get("incumbent_mean_abs_diff") is not None:
                f.write(f"| Mean abs diff vs incumbent | {report['incumbent_mean_abs_diff']:.4f} |\n")
            f.write("\n" + "\n".join(f"- `{t['service']}`: {t['weight']}%" for t in targets) + "\n")

    print("\nNext: apply_gateway_config.py applies the split. Edit gateway-config.yml to 100% and merge to cut over.")
    session.close()


if __name__ == "__main__":
    main()
