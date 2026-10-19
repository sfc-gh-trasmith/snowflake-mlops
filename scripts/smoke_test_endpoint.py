"""Post-deploy smoke test of the PROD real-time endpoint over HTTPS REST.

Exercises the same path a client uses: the gateway's public URL, with PAT
authentication and a dataframe_split body. Also calls each service behind the
gateway directly, so a 20% canary is checked even if the gateway's random split
happens to route every gateway request to the incumbent.

Checks, per URL:
  - HTTP 200 and one prediction per input row
  - outputs are valid for the task type (serving.signature.validate_outputs)
  - parity with warehouse scoring (ModelVersion.run) of the same version

Exits non-zero on any failure; deploy-prod then rolls PROD back.

Auth: uses SNOWFLAKE_PAT if set. Otherwise it issues a 1-day PAT for the
current user (restricted to the current role), uses it, and removes it. The
ingress rejects connector session tokens, so a PAT is required.

Usage:
  uv run python scripts/smoke_test_endpoint.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from apply_gateway_config import GATEWAY_NAME, get_current_gateway_targets
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

SMOKE_ROWS = int(REALTIME_CONFIG.get("smoke_test_rows", 10))
PARITY_TOLERANCE = 1e-4
REQUEST_TIMEOUT = 60
# A new gateway's ingress URL reads "provisioning..." for a few minutes, and
# the first requests after a gateway change can 5xx while routes propagate.
URL_WAIT_SECONDS = 600
RETRY_SECONDS = 180


def method_path(function_name: str) -> str:
    """Model method -> URL path (predict_proba -> /predict-proba)."""
    return "/" + function_name.replace("_", "-")


def rest_body(X: pd.DataFrame) -> dict:
    """dataframe_split payload; pandas serialization keeps types consistent."""
    return {"dataframe_split": json.loads(X.to_json(orient="split"))}


def parse_response(payload: dict, n_rows: int, columns: list[str]) -> np.ndarray:
    """{"data": [[row_index, {col: value}], ...]} -> (n_rows, n_cols) array in input order."""
    rows = sorted(payload.get("data") or [], key=lambda r: r[0])
    if len(rows) != n_rows:
        raise ValueError(f"expected {n_rows} predictions, got {len(rows)}")
    return np.array([[float(r[1][c]) for c in columns] for r in rows], dtype=float)


def check_outputs(rest_out: np.ndarray, warehouse_out: np.ndarray) -> list[str]:
    """Problems with REST outputs (empty = pass)."""
    from serving.signature import validate_outputs

    problems = validate_outputs(TASK_TYPE, rest_out)
    if rest_out.shape != warehouse_out.shape:
        return problems + [f"shape {rest_out.shape} != warehouse {warehouse_out.shape}"]
    diff = float(np.nanmax(np.abs(rest_out - warehouse_out)))
    if diff > PARITY_TOLERANCE:
        problems.append(f"REST vs warehouse outputs differ by {diff:.6f}")
    return problems


def _ingress_host(session, sql: str, column: str, where=None) -> str:
    deadline = time.time() + URL_WAIT_SECONDS
    while True:
        rows = session.sql(sql).collect()
        row = next((r for r in rows if where is None or where(r)), None)
        host = (row[column] if row else "") or ""
        if host and "provisioning" not in host.lower() and " " not in host:
            return host
        if time.time() > deadline:
            raise RuntimeError(f"No ingress URL after {URL_WAIT_SECONDS}s: {host!r}")
        print(f"  Waiting for ingress URL ({host or 'none'})...")
        time.sleep(15)


def post(url: str, token: str, X: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, float]:
    """POST with retries for transient 5xx/connection errors. Returns (outputs, latency_ms)."""
    import requests

    headers = {"Authorization": f'Snowflake Token="{token}"', "Content-Type": "application/json"}
    deadline = time.time() + RETRY_SECONDS
    while True:
        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=rest_body(X), timeout=REQUEST_TIMEOUT)
            latency = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                return parse_response(resp.json(), len(X), columns), latency
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            retryable = resp.status_code >= 500
        except requests.RequestException as e:
            error, retryable = f"{type(e).__name__}: {e}", True
        if not retryable or time.time() > deadline:
            hint = " (404 = wrong URL or the role lacks endpoint access)" if "HTTP 404" in error else ""
            raise RuntimeError(f"{url}: {error}{hint}")
        print(f"  {error} - retrying")
        time.sleep(10)


class EphemeralPAT:
    """A short-lived PAT for the current user, removed on exit."""

    def __init__(self, session):
        self.session = session
        self.name = f"MLOPS_SMOKE_{uuid.uuid4().hex[:8].upper()}"
        self.token = None

    def __enter__(self) -> str:
        if os.getenv("SNOWFLAKE_PAT"):
            return os.environ["SNOWFLAKE_PAT"]
        role = self.session.get_current_role().strip('"')
        row = self.session.sql(
            f"ALTER USER ADD PROGRAMMATIC ACCESS TOKEN {self.name} "
            f"ROLE_RESTRICTION = '{role}' DAYS_TO_EXPIRY = 1 "
            "COMMENT = 'snowflake-mlops post-deploy smoke test (auto-removed)'"
        ).collect()[0]
        self.token = row["token_secret"]
        if os.getenv("GITHUB_ACTIONS"):
            print(f"::add-mask::{self.token}")
        print(f"  Issued temporary PAT {self.name} (role {role}, expires in 1 day)")
        return self.token

    def __exit__(self, *exc):
        if self.token:
            self.session.sql(f"ALTER USER REMOVE PROGRAMMATIC ACCESS TOKEN {self.name}").collect()
            print(f"  Removed temporary PAT {self.name}")


def main():
    from serving.signature import cast_pandas, input_features, output_columns, run_outputs
    from snowflake.ml.registry import Registry

    print("=" * 60)
    print("PROD REST SMOKE TEST")
    print("=" * 60)
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()

    targets = get_current_gateway_targets(session, PROD_DATABASE, PROD_SCHEMA, GATEWAY_NAME)
    if not targets:
        sys.exit(f"Gateway {GATEWAY_NAME} has no targets; nothing to test")
    model = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA).get_model(MODEL_NAME)

    path = method_path(PREDICT_FUNCTION)
    gateway_host = _ingress_host(session, f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}", "ingress_url")
    checks = []
    for t in targets:
        version = t["service"].rsplit("_SERVICE_", 1)[-1]
        mv = model.version(version)
        host = _ingress_host(
            session,
            f"SHOW ENDPOINTS IN SERVICE {PROD_DATABASE}.{PROD_SCHEMA}.{t['service']}",
            "ingress_url",
            where=lambda r: r["name"] == "inference",
        )
        checks.append({"name": f"service {t['service']} ({t['weight']}%)", "url": f"https://{host}{path}", "mv": mv})
    # All targets share one signature contract (the gateway splits one URL across them).
    checks.insert(0, {"name": f"gateway {GATEWAY_NAME}", "url": f"https://{gateway_host}{path}", "mv": checks[0]["mv"]})

    feats = input_features(checks[0]["mv"], PREDICT_FUNCTION)
    columns = output_columns(checks[0]["mv"], PREDICT_FUNCTION)
    sample = session.table(batch_input_table(PROD_DATABASE, PROD_SCHEMA)).select(*feats).limit(SMOKE_ROWS).to_pandas()
    if sample.empty:
        sys.exit("Smoke test: batch input table returned no rows")
    X = cast_pandas(sample, checks[0]["mv"], PREDICT_FUNCTION)
    expected = {id(c["mv"]): run_outputs(c["mv"], sample, PREDICT_FUNCTION) for c in checks}

    results = []
    with EphemeralPAT(session) as token:
        for c in checks:
            print(f"\n  {c['name']}: POST {c['url']}")
            try:
                out, latency = post(c["url"], token, X, columns)
                if c["name"].startswith("gateway"):
                    # Each gateway request may hit any target: accept parity with any of them.
                    problems = min((check_outputs(out, expected[id(o["mv"])]) for o in checks[1:]), key=len)
                else:
                    problems = check_outputs(out, expected[id(c["mv"])])
            except Exception as e:
                problems, latency = [str(e)], None
            results.append({"check": c["name"], "latency_ms": latency, "problems": problems})
            print(f"    {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}")

    session.close()
    failed = [r for r in results if r["problems"]]

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(
                f"\n## REST Smoke Test ({len(X)} rows, `{path}`)\n\n| Endpoint | Result | Latency |\n|---|---|---|\n"
            )
            for r in results:
                status = "pass" if not r["problems"] else "**FAIL**: " + "; ".join(r["problems"])
                latency = f"{r['latency_ms']:.0f} ms" if r["latency_ms"] is not None else "-"
                f.write(f"| {r['check']} | {status} | {latency} |\n")

    if failed:
        sys.exit(f"\nSMOKE TEST FAILED: {len(failed)} of {len(results)} endpoint checks")
    print(f"\nSMOKE TEST PASSED: {len(results)} endpoint checks")


if __name__ == "__main__":
    main()
