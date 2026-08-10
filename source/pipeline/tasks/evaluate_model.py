"""Model Evaluation task — reads metrics from training step and validates.

This is a lightweight step that confirms training completed and metrics are available.
The actual quality gate decision happens in GH Actions (human-in-the-loop).

Writes final pipeline status to PIPELINE_RESULTS.
"""

import json
import os
import sys

from snowflake.snowpark import Session


def main():
    session = Session.builder.getOrCreate()

    database = os.getenv("PIPELINE_DATABASE", "SNOW_MLOPS_STAGE")
    schema = os.getenv("PIPELINE_SCHEMA", "ML")
    warehouse = os.getenv("PIPELINE_WAREHOUSE", f"{database}_WH")

    session.sql(f"USE WAREHOUSE {warehouse}").collect()

    print("Evaluation: reading training metrics...")
    rows = session.sql(f"""
        SELECT RESULT FROM {database}.{schema}.PIPELINE_RESULTS
        WHERE STEP = 'training' AND STATUS = 'SUCCESS'
        ORDER BY CREATED_AT DESC LIMIT 1
    """).collect()

    if not rows:
        error_result = {"status": "error", "step": "evaluation", "message": "No training results found"}
        session.sql(f"""
            INSERT INTO {database}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
            VALUES ('evaluation', 'ERROR', PARSE_JSON('{json.dumps(error_result)}'), CURRENT_TIMESTAMP())
        """).collect()
        print("ERROR: No training results found in PIPELINE_RESULTS")
        session.close()
        sys.exit(1)

    training_result = json.loads(rows[0]["RESULT"])
    metrics = training_result.get("metrics", {})

    print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    print(f"  Precision: {metrics.get('precision', 'N/A')}")
    print(f"  Recall: {metrics.get('recall', 'N/A')}")
    print(f"  F1: {metrics.get('f1', 'N/A')}")

    # Mark pipeline as complete (ready for GH Actions to pick up)
    result = {
        "status": "success",
        "step": "evaluation",
        "metrics": metrics,
        "pipeline_status": "READY_FOR_REVIEW",
    }
    result_json = json.dumps(result).replace("'", "''")
    session.sql(f"""
        INSERT INTO {database}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
        VALUES ('evaluation', 'SUCCESS', PARSE_JSON('{result_json}'), CURRENT_TIMESTAMP())
    """).collect()

    print("Evaluation complete. Pipeline ready for quality gate review.")
    session.close()


if __name__ == "__main__":
    main()
