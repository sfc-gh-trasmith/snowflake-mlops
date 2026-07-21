"""Batch Inference CI entry point.

Scores the feature view table using the latest registered model and validates
predictions are sane. Used in both STAGE and PROD CI workflows.

Exit code 1 if validation fails (null predictions or degenerate probabilities).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

# Environment-aware configuration
DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOW_MLOPS_DEV")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", f"{DATABASE}_WH")

FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "V1"
INPUT_TABLE = f'{DATABASE}.{SCHEMA}."{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"'
OUTPUT_TABLE = f"{DATABASE}.{SCHEMA}.BATCH_PREDICTIONS"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"


def write_batch_summary(result: dict, validation: dict):
    """Write batch inference results to GitHub Actions Job Summary."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    status = "PASSED" if validation["passed"] else "FAILED"
    md = f"""## Batch Inference — {status}

| Metric | Value |
|--------|-------|
| Rows scored | {result['row_count']:,} |
| Model version | `{result['model_version']}` |
| Null predictions | {validation['null_predictions']} |
| Avg probability sum | {validation['avg_probability_sum']:.6f} |
| Max deviation from 1.0 | {validation['max_deviation_from_1']:.6f} |
| Output table | `{result['output_table']}` |
"""
    with open(summary_path, "a") as f:
        f.write(md)
    print("  Batch summary written to GITHUB_STEP_SUMMARY")


def main():
    print("=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    from serving.batch_inference import run_batch_inference, validate_predictions

    # Step 1: Run batch scoring
    print(f"\n[1/2] Scoring {INPUT_TABLE}...")
    result = run_batch_inference(
        session=session,
        input_table=INPUT_TABLE,
        output_table=OUTPUT_TABLE,
        model_name=MODEL_NAME,
    )

    # Step 2: Validate predictions
    print("\n[2/2] Validating predictions...")
    validation = validate_predictions(session, OUTPUT_TABLE)

    if validation["passed"]:
        print("  Validation PASSED")
        print(f"    Rows: {validation['total_rows']:,}")
        print(f"    Null predictions: {validation['null_predictions']}")
        print(f"    Max deviation: {validation['max_deviation_from_1']:.6f}")
    else:
        print("  Validation FAILED")
        print(f"    Null predictions: {validation['null_predictions']}")
        print(f"    Max deviation: {validation['max_deviation_from_1']:.6f}")

    # Write summary for CI
    write_batch_summary(result, validation)

    print("\n" + "=" * 60)
    print(f"BATCH INFERENCE COMPLETE — {result['row_count']:,} rows scored")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Model: {MODEL_NAME} ({result['model_version']})")
    print("=" * 60)

    session.close()

    if not validation["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
