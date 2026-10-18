"""Batch inference pipeline.

A scheduled Task that scores the batch input table with the model's default
version and APPENDS predictions (serving/batch_inference.py). Runs on a
warehouse as an anonymous stored procedure importing the same source.zip
pattern as the training DAG.

Usage:
    # Deploy the scheduled scoring task (serving.batch.schedule) and resume it
    python source/pipeline/batch_inference_dag.py --deploy  --env prod
    # Run it now
    python source/pipeline/batch_inference_dag.py --execute --env prod
    python source/pipeline/batch_inference_dag.py --status  --env prod

    # CI validation: score into <output_table>_VALIDATION (overwrite), check
    # outputs, write a Job Summary. Exit 1 if validation fails. Optionally pin
    # the version under test with MODEL_VERSION=V3.
    python source/pipeline/batch_inference_dag.py --validate --env stage
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_CONFIG,
    MODEL_NAME,
    batch_input_table,
    batch_output_table,
    resolve_env_config,
)
from pipeline.train_dag import SP_PACKAGES, build_source_zip, runtime_env, sproc_definition
from snowpark_session import create_snowpark_session

DAG_NAME = f"{MODEL_NAME}_BATCH_INFERENCE"
TASK_NAME = "SCORE"
VALIDATION_SUFFIX = "_VALIDATION"


def validation_table(database: str, schema: str) -> str:
    return batch_output_table(database, schema) + VALIDATION_SUFFIX


def _parse_schedule():
    schedule = BATCH_CONFIG.get("schedule") or ""
    if not schedule:
        return None
    from snowflake.core.task import Cron

    tokens = schedule.replace("USING CRON ", "").strip().split()
    return Cron(" ".join(tokens[:-1]), tokens[-1])


def deploy(env: str):
    """Deploy the scheduled batch scoring task and resume it."""
    from snowflake.core import Root
    from snowflake.core._common import CreateMode
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask

    cfg = resolve_env_config(env)
    db, schema, wh = cfg["database"], cfg["schema"], cfg["warehouse"]
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    stage = f"@{db}.{schema}.PIPELINE_STAGE"
    session.sql(f"CREATE STAGE IF NOT EXISTS {db}.{schema}.PIPELINE_STAGE").collect()
    print(f"Deploying batch inference task: {db}.{schema}.{DAG_NAME}")

    with tempfile.TemporaryDirectory() as tmp:
        source_zip = build_source_zip(Path(tmp))
        session.file.put(str(source_zip), f"{stage}/batch_inference/", auto_compress=False, overwrite=True)

    definition = sproc_definition(
        "serving.batch_inference",
        runtime_env(cfg),
        imports_clause=f"'{stage}/batch_inference/source.zip'",
        packages_clause=",".join(f"'{p}'" for p in SP_PACKAGES),
    )
    schedule = _parse_schedule()
    with DAG(DAG_NAME, warehouse=wh, schedule=schedule, stage_location=stage, use_func_return_value=True) as dag:
        DAGTask(TASK_NAME, definition=definition, warehouse=wh)

    DAGOperation(Root(session).databases[db].schemas[schema]).deploy(dag, mode=CreateMode.or_replace)
    if schedule:
        session.sql(f"ALTER TASK {db}.{schema}.{DAG_NAME} RESUME").collect()
        print(f"  Scheduled: {BATCH_CONFIG['schedule']} (resumed)")
    print(f"  Input: {batch_input_table(db, schema)}")
    print(f"  Output (append): {batch_output_table(db, schema)}")
    session.close()


def execute(env: str):
    from snowflake.core import Root

    cfg = resolve_env_config(env)
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {cfg['warehouse']}").collect()
    Root(session).databases[cfg["database"]].schemas[cfg["schema"]].tasks[DAG_NAME].execute()
    print(f"Executed: {cfg['database']}.{cfg['schema']}.{DAG_NAME}")
    session.close()


def status(env: str):
    cfg = resolve_env_config(env)
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {cfg['warehouse']}").collect()
    rows = session.sql(f"""
        SELECT NAME, STATE, SCHEDULED_TIME, RETURN_VALUE, ERROR_MESSAGE
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
            SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP()), RESULT_LIMIT => 1000))
        WHERE DATABASE_NAME = '{cfg["database"]}' AND NAME LIKE '{DAG_NAME}%'
        ORDER BY SCHEDULED_TIME DESC
    """).collect()
    if not rows:
        print("  No batch inference runs in the last 7 days.")
    for r in rows:
        print(f"  [{str(r['SCHEDULED_TIME'])[:19]}] {r['NAME']}: {r['STATE']} {(r['ERROR_MESSAGE'] or '')[:100]}")
    session.close()


def _write_summary(result: dict, validation: dict):
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    dev = validation.get("max_deviation_from_1")
    with open(summary_path, "a") as f:
        f.write(f"\n## Batch Inference Validation — {'PASSED' if validation['passed'] else 'FAILED'}\n\n")
        f.write("| Check | Value |\n|-------|-------|\n")
        f.write(f"| Model version | `{result['model_version']}` |\n")
        f.write(f"| Rows scored | {result['row_count']:,} |\n")
        f.write(f"| Null predictions | {validation['null_predictions']} |\n")
        if dev is not None:
            f.write(f"| Max probability-sum deviation | {dev:.6f} |\n")
        f.write(f"| Output table | `{result['output_table']}` |\n")
        for p in validation["problems"]:
            f.write(f"\n- FAILED: {p}\n")


def validate(env: str) -> bool:
    """CI check: score with the version under test into a validation table."""
    from feature_engineering.feature_views import register_feature_views
    from serving.batch_inference import run_batch_inference, validate_predictions

    cfg = resolve_env_config(env)
    db, schema, wh = cfg["database"], cfg["schema"], cfg["warehouse"]
    version = os.getenv("MODEL_VERSION") or None

    print("=" * 60)
    print(f"BATCH INFERENCE VALIDATION ({db}.{schema})")
    print("=" * 60)
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print("\n[1/3] Ensuring Feature Views exist (idempotent)...")
    register_feature_views(session=session, database=db, schema=schema, warehouse=wh)

    print("\n[2/3] Scoring...")
    result = run_batch_inference(
        session,
        input_table=batch_input_table(db, schema),
        output_table=validation_table(db, schema),
        version=version,
        mode="overwrite",
        database=db,
        schema=schema,
    )

    print("\n[3/3] Validating predictions...")
    validation = validate_predictions(session, result)
    print(f"  {'PASSED' if validation['passed'] else 'FAILED'}: {validation['total_rows']:,} rows")
    for p in validation["problems"]:
        print(f"    - {p}")
    _write_summary(result, validation)
    session.close()
    return validation["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference Task")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--deploy", action="store_true", help="Deploy and resume the scheduled scoring task")
    group.add_argument("--execute", action="store_true", help="Run the scoring task now")
    group.add_argument("--status", action="store_true", help="Show recent runs")
    group.add_argument("--validate", action="store_true", help="CI: score into a validation table and check outputs")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "dev"), choices=["dev", "stage", "prod"])
    args = parser.parse_args()

    if args.deploy:
        deploy(args.env)
    elif args.execute:
        execute(args.env)
    elif args.status:
        status(args.env)
    elif args.validate:
        sys.exit(0 if validate(args.env) else 1)
    else:
        parser.print_help()
