"""Wait for Snowflake Task DAG completion.

Polls TASK_HISTORY until all tasks in the pipeline have completed (SUCCESS or FAILED).
Outputs metrics from PIPELINE_RESULTS for the GH Actions Job Summary.

Exit code 1 if any task failed.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOW_MLOPS_STAGE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", f"{DATABASE}_WH")

POLL_INTERVAL = 30  # seconds
MAX_WAIT = 3600  # 1 hour


def main():
    print("=" * 60)
    print("WAITING FOR TASK DAG COMPLETION")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    expected_tasks = {"ML_TRAINING_PIPELINE$FEATURE_ENG", "ML_TRAINING_PIPELINE$TRAIN_MODEL", "ML_TRAINING_PIPELINE$EVALUATE"}
    start = time.time()

    while time.time() - start < MAX_WAIT:
        rows = session.sql(f"""
            SELECT NAME, STATE, ERROR_MESSAGE
            FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP()),
                RESULT_LIMIT => 20
            ))
            WHERE DATABASE_NAME = '{DATABASE}'
            ORDER BY SCHEDULED_TIME DESC
        """).collect()

        completed = {}
        for row in rows:
            name = row["NAME"]
            state = row["STATE"]
            if name in expected_tasks and name not in completed:
                completed[name] = {"state": state, "error": row["ERROR_MESSAGE"]}

        # Check for failures
        failed = {k: v for k, v in completed.items() if v["state"] == "FAILED"}
        if failed:
            print("\nTASK FAILED:")
            for name, info in failed.items():
                print(f"  {name}: {info['error']}")
            session.close()
            sys.exit(1)

        # Check if all expected tasks succeeded
        succeeded = {k for k, v in completed.items() if v["state"] == "SUCCEEDED"}
        if expected_tasks.issubset(succeeded):
            print("\nAll tasks completed successfully!")
            break

        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] Completed: {len(succeeded)}/{len(expected_tasks)} — waiting...")
        time.sleep(POLL_INTERVAL)
    else:
        print(f"\nTIMEOUT: Tasks did not complete within {MAX_WAIT}s")
        session.close()
        sys.exit(1)

    # Read metrics from PIPELINE_RESULTS
    print("\nReading pipeline results...")
    results = session.sql(f"""
        SELECT STEP, STATUS, RESULT FROM {DATABASE}.{SCHEMA}.PIPELINE_RESULTS
        ORDER BY CREATED_AT DESC
    """).collect()

    metrics = {}
    for row in results:
        if row["STEP"] == "evaluation" and row["STATUS"] == "SUCCESS":
            result_data = json.loads(row["RESULT"])
            metrics = result_data.get("metrics", {})
            break

    if metrics:
        print(f"\n  AUC-ROC:    {metrics.get('auc_roc', 'N/A')}")
        print(f"  Precision:  {metrics.get('precision', 'N/A')}")
        print(f"  Recall:     {metrics.get('recall', 'N/A')}")
        print(f"  F1:         {metrics.get('f1', 'N/A')}")

    # Write to Job Summary
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path and metrics:
        with open(summary_path, "a") as f:
            f.write("## ML Training Pipeline Results\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"| {k} | {v:.4f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")
        print("  Metrics written to Job Summary.")

    # Write metrics to file for quality gate step
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    print(f"  Metrics saved to {metrics_file}")

    session.close()
    print("\n" + "=" * 60)
    print("TASK DAG COMPLETE — Ready for quality gate review")
    print("=" * 60)


if __name__ == "__main__":
    main()
