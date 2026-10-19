"""Wait for the training Task DAG to finish and export the evaluation result.

Polls TASK_HISTORY until every step of the training DAG has completed, then
writes the EVALUATE step's result (model version, metrics, gate decision) to
METRICS_FILE for downstream CI steps (utils/read_candidate.py).

Exit code 1 if any task failed (including a quality gate rejection).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import env_database
from pipeline.train_dag import DAG_NAME, STEP_ORDER
from snowpark_session import create_snowpark_session

DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("stage"))
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", f"{DATABASE}_WH")

POLL_INTERVAL = 30  # seconds
MAX_WAIT = 3600  # 1 hour


def _latest_evaluation(session, since=None) -> dict:
    """Evaluation written by this run (on or after the root task's scheduled time).

    Without the time bound, a run that failed before EVALUATE would report the
    previous run's candidate. CREATED_AT is TIMESTAMP_NTZ (session-local), so
    this assumes the pipeline and CI sessions share the account TIMEZONE.
    """
    where = f"AND CREATED_AT >= '{since}'::TIMESTAMP_LTZ" if since else ""
    rows = session.sql(f"""
        SELECT RESULT FROM {DATABASE}.{SCHEMA}.PIPELINE_RESULTS
        WHERE STEP = 'evaluation' {where}
        ORDER BY CREATED_AT DESC LIMIT 1
    """).collect()
    return json.loads(rows[0]["RESULT"]) if rows else {}


def _task_states(session, since) -> dict:
    window = (
        f"SCHEDULED_TIME_RANGE_START => '{since}'::TIMESTAMP_LTZ"
        if since
        else "SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP())"
    )
    rows = session.sql(f"""
        SELECT NAME, STATE, ERROR_MESSAGE
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY({window}, RESULT_LIMIT => 1000))
        WHERE DATABASE_NAME = '{DATABASE}'
        ORDER BY SCHEDULED_TIME DESC
    """).collect()
    states = {}
    for row in rows:
        states.setdefault(row["NAME"], {"state": row["STATE"], "error": row["ERROR_MESSAGE"]})
    return states


def main():
    print("=" * 60)
    print("WAITING FOR TASK DAG COMPLETION")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    expected = {f"{DAG_NAME}${step}" for step in STEP_ORDER}
    # Filter after the table function: RESULT_LIMIT applies before WHERE, and
    # the same task name exists in every environment's database (plus a future
    # SCHEDULED row), so a small limit can miss this database's latest run.
    root = session.sql(f"""
        SELECT SCHEDULED_TIME
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(TASK_NAME => '{DAG_NAME}', RESULT_LIMIT => 1000))
        WHERE DATABASE_NAME = '{DATABASE}' AND STATE <> 'SCHEDULED'
        ORDER BY SCHEDULED_TIME DESC
        LIMIT 1
    """).collect()
    since = root[0]["SCHEDULED_TIME"] if root else None
    print(f"  Root task executed at: {since or 'not found (looking at last hour)'}")

    start = time.time()
    failed = {}
    while time.time() - start < MAX_WAIT:
        states = {k: v for k, v in _task_states(session, since).items() if k in expected}
        failed = {k: v for k, v in states.items() if v["state"] == "FAILED"}
        succeeded = {k for k, v in states.items() if v["state"] == "SUCCEEDED"}
        if failed or expected.issubset(succeeded):
            break
        print(f"  [{int(time.time() - start)}s] Completed: {len(succeeded)}/{len(expected)} — waiting...")
        time.sleep(POLL_INTERVAL)
    else:
        print(f"\nTIMEOUT: Tasks did not complete within {MAX_WAIT}s")
        session.close()
        sys.exit(1)

    evaluation = _latest_evaluation(session, since)
    if failed:
        # Lets the CI summary say which step failed when EVALUATE never ran.
        evaluation.setdefault("failed_tasks", {k: v["error"] for k, v in failed.items()})
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(evaluation, f)
    session.close()

    if failed:
        print("\nTASK FAILED:")
        for name, info in failed.items():
            print(f"  {name}: {info['error']}")
        if evaluation.get("gate_passed") is False:
            print("\nQuality gate rejected the candidate:")
            for failure in evaluation.get("gate_failures", []):
                print(f"  - {failure}")
        sys.exit(1)

    print("\nAll tasks completed successfully!")
    print(f"  Candidate: {evaluation.get('model_version')}")
    for k, v in evaluation.get("metrics", {}).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Evaluation result saved to {metrics_file}")
    print("\n" + "=" * 60)
    print("TASK DAG COMPLETE — quality gate passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
