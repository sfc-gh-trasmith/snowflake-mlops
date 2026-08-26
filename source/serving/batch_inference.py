"""Batch inference: score a table using the registered model on warehouse compute.

Uses model.run() from the Model Registry — no SPCS service needed.
Runs on the warehouse, suitable for bulk scoring (daily/hourly).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE, MODEL_NAME, SCHEMA


def run_batch_inference(
    session,
    input_table: str,
    output_table: str,
    model_name: str = MODEL_NAME,
    version: str | None = None,
) -> dict:
    """Score an entire table using the registered model.

    Args:
        session: Active Snowpark session
        input_table: Fully-qualified input table with feature columns
        output_table: Where to write predictions (overwritten each run)
        model_name: Model to use from the registry
        version: Specific version, or None for default

    Returns:
        dict with row_count and output_table
    """
    from snowflake.ml.registry import Registry

    db = os.getenv("SNOWFLAKE_DATABASE", DATABASE)
    schema = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)

    reg = Registry(session=session, database_name=db, schema_name=schema)
    model = reg.get_model(model_name)
    mv = model.version(version) if version else model.default

    print(f"  Model: {model_name} (version: {mv.version_name})")
    print(f"  Input: {input_table}")

    input_df = session.table(input_table)
    row_count = input_df.count()
    print(f"  Rows to score: {row_count:,}")

    predictions_df = mv.run(input_df, function_name="predict_proba")

    # Add timestamp for model monitoring
    from snowflake.snowpark.functions import current_timestamp

    predictions_df = predictions_df.with_column("PREDICTION_TS", current_timestamp())

    predictions_df.write.mode("overwrite").save_as_table(output_table)
    print(f"  Output written to: {output_table}")

    return {
        "row_count": row_count,
        "output_table": output_table,
        "model_version": mv.version_name,
    }


def validate_predictions(session, output_table: str) -> dict:
    """Validate batch predictions are sane (no nulls, probabilities sum to ~1)."""
    from snowflake.snowpark.functions import abs as sf_abs
    from snowflake.snowpark.functions import avg, col, count, when
    from snowflake.snowpark.functions import max as sf_max

    df = session.table(output_table)

    # Column names from model.run() are lowercase quoted identifiers
    col_0 = '"output_feature_0"'
    col_1 = '"output_feature_1"'

    # Push all validation into Snowflake — no client-side data pull
    stats = df.select(
        count("*").alias("TOTAL"),
        count(when(col(col_0).is_null() | col(col_1).is_null(), 1)).alias("NULLS"),
        avg(col(col_0) + col(col_1)).alias("AVG_SUM"),
        sf_max(sf_abs(col(col_0) + col(col_1) - 1.0)).alias("MAX_DEV"),
    ).collect()[0]

    validation = {
        "total_rows": stats["TOTAL"],
        "null_predictions": stats["NULLS"],
        "avg_probability_sum": float(stats["AVG_SUM"]),
        "max_deviation_from_1": float(stats["MAX_DEV"]),
        "passed": stats["NULLS"] == 0 and float(stats["MAX_DEV"]) < 0.01,
    }

    return validation
