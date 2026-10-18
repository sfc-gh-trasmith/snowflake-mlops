"""Model-signature helpers shared by batch scoring, real-time checks, and training.

The Model Registry validates input types strictly against the signature
captured at log_model() time. These helpers derive casts and output columns
from that signature, so no feature names or model types are hardcoded in the
serving path.
"""

import numpy as np


def _function_signature(mv, function_name: str):
    for fn in mv.show_functions():
        if fn["name"].upper() == function_name.upper():
            return fn["signature"]
    available = [fn["name"] for fn in mv.show_functions()]
    raise ValueError(f"Model version {mv.version_name} has no function '{function_name}'. Available: {available}")


def input_features(mv, function_name: str) -> list[str]:
    """Names of the input features the function expects, in signature order."""
    return [spec.name for spec in _function_signature(mv, function_name).inputs]


def output_columns(mv, function_name: str) -> list[str]:
    """Output column names produced by ModelVersion.run() for the function."""
    return [spec.name for spec in _function_signature(mv, function_name).outputs]


def cast_snowpark(df, mv, function_name: str):
    """Cast a Snowpark DataFrame's feature columns to the signature's types."""
    from snowflake.snowpark.functions import col

    for spec in _function_signature(mv, function_name).inputs:
        if spec.name in df.columns:
            df = df.with_column(spec.name, col(spec.name).cast(spec.as_snowpark_type()))
    return df


def cast_pandas(df, mv, function_name: str):
    """Cast a pandas DataFrame's feature columns to the signature's types."""
    import pandas as pd

    df = df.copy()
    for spec in _function_signature(mv, function_name).inputs:
        if spec.name in df.columns:
            dtype = spec.as_dtype()  # numpy dtype, or a pandas extension dtype (e.g. Int64)
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                # Snowflake NUMBER columns arrive as Decimal objects
                df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
            if pd.api.types.is_integer_dtype(dtype):
                df[spec.name] = df[spec.name].fillna(0).round()
            df[spec.name] = df[spec.name].astype(dtype)
    return df


def run_outputs(mv, X, function_name: str, **run_kwargs) -> np.ndarray:
    """Run a registered model version on pandas input; return outputs as an array.

    Used to score the champion on the candidate's holdout, and by real-time
    health checks (pass service_name=... to target an inference service).
    """
    feats = input_features(mv, function_name)
    X_cast = cast_pandas(X[feats], mv, function_name)
    result = mv.run(X_cast, function_name=function_name, **run_kwargs)
    return result[output_columns(mv, function_name)].to_numpy(dtype=float)


def validate_outputs(task_type: str, outputs: np.ndarray, tolerance: float = 0.01) -> list[str]:
    """Sanity-check prediction outputs. Returns a list of problems (empty = OK)."""
    out = np.asarray(outputs, dtype=float)
    out = out.reshape(-1, 1) if out.ndim == 1 else out
    problems = []
    if out.size == 0:
        return ["no predictions returned"]
    if np.isnan(out).any():
        problems.append(f"{int(np.isnan(out).any(axis=1).sum())} rows with null predictions")
    if not np.isfinite(out[~np.isnan(out)]).all():
        problems.append("non-finite predictions")
    if task_type in ("binary_classification", "multiclass") and out.shape[1] > 1:
        clean = out[~np.isnan(out).any(axis=1)]
        if ((clean < -tolerance) | (clean > 1 + tolerance)).any():
            problems.append("probabilities outside [0, 1]")
        max_dev = float(np.abs(clean.sum(axis=1) - 1.0).max()) if len(clean) else 0.0
        if max_dev > tolerance:
            problems.append(f"probabilities do not sum to 1 (max deviation {max_dev:.4f})")
    return problems


def validate_outputs_sql(df, task_type: str, columns: list[str], tolerance: float = 0.01) -> dict:
    """Validate predictions in a Snowpark DataFrame, computed entirely in Snowflake."""
    from snowflake.snowpark import functions as F

    cols = [F.col(f'"{c}"') if c != c.upper() else F.col(c) for c in columns]
    any_null = cols[0].is_null()
    for c in cols[1:]:
        any_null = any_null | c.is_null()

    aggs = [F.count(F.lit(1)).alias("TOTAL"), F.count(F.when(any_null, 1)).alias("NULLS")]
    is_proba = task_type in ("binary_classification", "multiclass") and len(cols) > 1
    if is_proba:
        total = cols[0]
        for c in cols[1:]:
            total = total + c
        aggs.append(F.max(F.abs(total - 1.0)).alias("MAX_DEV"))
    stats = df.select(*aggs).collect()[0]

    problems = []
    if stats["TOTAL"] == 0:
        problems.append("no predictions written")
    if stats["NULLS"]:
        problems.append(f"{stats['NULLS']} rows with null predictions")
    max_dev = float(stats["MAX_DEV"] or 0.0) if is_proba else 0.0
    if is_proba and max_dev > tolerance:
        problems.append(f"probabilities do not sum to 1 (max deviation {max_dev:.4f})")
    return {
        "total_rows": stats["TOTAL"],
        "null_predictions": stats["NULLS"],
        "max_deviation_from_1": max_dev if is_proba else None,
        "problems": problems,
        "passed": not problems,
    }
