"""Batch inference: score a table with a registered model on warehouse compute.

Uses ModelVersion.run() from the Model Registry (no inference service needed).
Input casting and output validation are derived from the model signature, so
this works for any registered model.

Every run APPENDS rows tagged with MODEL_VERSION, SCORED_AT and RUN_ID, which
gives the model monitor a prediction history and makes every score traceable
to the version that produced it.

run(session) is the scheduled Task entrypoint (pipeline/batch_inference_dag.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE,
    MODEL_NAME,
    PREDICT_FUNCTION,
    SCHEMA,
    TASK_TYPE,
    WAREHOUSE,
    batch_input_table,
    batch_output_table,
)


def run_batch_inference(
    session,
    input_table: str,
    output_table: str,
    model_name: str = MODEL_NAME,
    version: str | None = None,
    mode: str = "append",
    database: str = DATABASE,
    schema: str = SCHEMA,
) -> dict:
    """Score input_table and write predictions to output_table.

    Args:
        version: model version to use; None = the model's default version
        mode: "append" (scheduled scoring) or "overwrite" (CI validation tables)

    Returns dict with run_id, row_count, output_table, model_version, and the
    prediction output columns.
    """
    import uuid

    from serving.signature import cast_snowpark, input_features, output_columns
    from snowflake.ml.registry import Registry
    from snowflake.snowpark import functions as F

    reg = Registry(session=session, database_name=database, schema_name=schema)
    model = reg.get_model(model_name)
    mv = model.version(version) if version else model.default
    run_id = str(uuid.uuid4())

    print(f"  Model: {model_name} (version: {mv.version_name}, function: {PREDICT_FUNCTION})")
    print(f"  Input: {input_table}")

    input_df = session.table(input_table)
    missing = [c for c in input_features(mv, PREDICT_FUNCTION) if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input table {input_table} is missing model features: {missing}")
    original_types = {f.name: f.datatype for f in input_df.schema.fields}
    row_count = input_df.count()
    print(f"  Rows to score: {row_count:,}")

    predictions = mv.run(cast_snowpark(input_df, mv, PREDICT_FUNCTION), function_name=PREDICT_FUNCTION)
    # Casting to the signature is only for scoring. Restore the input's types so
    # the appended history keeps one schema across model versions (signatures
    # can differ between versions).
    restore = [c for c in predictions.columns if c in original_types]
    predictions = predictions.with_columns(restore, [F.col(c).cast(original_types[c]) for c in restore])
    predictions = predictions.with_columns(
        ["MODEL_VERSION", "SCORED_AT", "RUN_ID"],
        [F.lit(mv.version_name), F.current_timestamp().cast("TIMESTAMP_NTZ"), F.lit(run_id)],
    )
    predictions.write.mode(mode).save_as_table(output_table, column_order="name")
    print(f"  Output {'appended to' if mode == 'append' else 'written to'}: {output_table}")

    return {
        "run_id": run_id,
        "row_count": row_count,
        "output_table": output_table,
        "model_version": mv.version_name,
        "output_columns": output_columns(mv, PREDICT_FUNCTION),
    }


def validate_predictions(session, result: dict) -> dict:
    """Validate the rows written by one scoring run (filtered by RUN_ID)."""
    from serving.signature import validate_outputs_sql
    from snowflake.snowpark import functions as F

    df = session.table(result["output_table"]).filter(F.col("RUN_ID") == result["run_id"])
    validation = validate_outputs_sql(df, TASK_TYPE, result["output_columns"])
    if validation["total_rows"] != result["row_count"]:
        validation["problems"].append(f"wrote {validation['total_rows']} rows, expected {result['row_count']}")
        validation["passed"] = False
    return validation


def run(session) -> dict:
    """Scheduled batch scoring: default model version, append, validate."""
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    result = run_batch_inference(
        session,
        input_table=batch_input_table(DATABASE, SCHEMA),
        output_table=batch_output_table(DATABASE, SCHEMA),
    )
    validation = validate_predictions(session, result)
    if not validation["passed"]:
        raise RuntimeError(f"Batch predictions failed validation: {validation['problems']}")
    return {"status": "success", "step": "batch_inference", **result, "validation": validation}
