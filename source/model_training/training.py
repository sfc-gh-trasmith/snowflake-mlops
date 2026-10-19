"""Model training step.

Builds a point-in-time correct dataset from the Feature Store, trains a model,
evaluates it on a grouped holdout, scores the current champion on the same
holdout, and registers the candidate in the Model Registry (linked to the
training Dataset and the Experiment Tracking run).

To bring your own model, replace the two example functions below:
  build_dataset(session) -> (Dataset, pandas.DataFrame)
  train_model(X_train, y_train) -> fitted model exposing model.predict_function

Everything else (holdout split, metrics, champion scoring, registration,
results contract) is model-agnostic.

Contract consumed by the EVALUATE step (evaluate/evaluation.py):
  a PIPELINE_RESULTS row with STEP='training', STATUS='SUCCESS' and RESULT
  {"model_version", "metrics", "champion": {...}, "feature_view", ...}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CHAMPION_CONFIG,
    DATABASE,
    EXPERIMENT_CONFIG,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    ID_COLUMNS,
    LABEL_COLUMN,
    MODEL_DEPS,
    MODEL_NAME,
    PREDICT_FUNCTION,
    SCHEMA,
    SOURCE_DATABASE,
    SOURCE_SCHEMA,
    TASK_TYPE,
    TRAINING_PARAMS,
    WAREHOUSE,
    env_database,
)
from model_training.metrics import compute_metrics

FEATURE_VIEW = f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"
DATASET_NAME = f"{MODEL_NAME}_TRAINING_DATA"
SPINE_TIMESTAMP_COL = "TIMESTAMP"
HOLDOUT_FOLDS = 5  # 1 of 5 grouped folds = 20% holdout
RANDOM_STATE = 42


# ─── Example-specific: replace these two for your use case ───────────────────


def dataset_version() -> str:
    """Unique Dataset version per run: <feature view version>_<UTC timestamp>."""
    from datetime import datetime, timezone

    return f"{FEATURE_VIEW_VERSION}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"


def build_dataset(session):
    """Generate the training Dataset from the Feature Store.

    Returns (Dataset, pandas DataFrame). The DataFrame must contain the
    ID_COLUMNS, LABEL_COLUMN, and feature columns; any other columns listed in
    non_feature_columns() are dropped before training.
    """
    from snowflake.ml.feature_store import CreationMode, FeatureStore
    from snowflake.snowpark import functions as F

    fs = FeatureStore(
        session=session,
        database=DATABASE,
        name=SCHEMA,
        default_warehouse=WAREHOUSE,
        creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
    )
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)

    spine = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS").select(
        *[F.col(c) for c in ID_COLUMNS], F.col(SPINE_TIMESTAMP_COL), F.col(LABEL_COLUMN)
    )

    # Each run takes a fresh point-in-time snapshot, so retraining actually
    # sees new data. The version name carries the feature view version, and
    # registry lineage links model -> Dataset version -> feature view version.
    dataset = fs.generate_dataset(
        name=DATASET_NAME,
        version=dataset_version(),
        spine_df=spine,
        features=[fv],
        spine_timestamp_col=SPINE_TIMESTAMP_COL,
        spine_label_cols=[LABEL_COLUMN],
        desc=f"Training dataset for {MODEL_NAME} from {FEATURE_VIEW}",
    )

    return dataset, dataset.read.to_pandas()


def train_model(X_train, y_train):
    """Fit the model. Must expose the configured predict function."""
    import xgboost as xgb

    model = xgb.XGBClassifier(**TRAINING_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


def non_feature_columns() -> list[str]:
    return [*ID_COLUMNS, LABEL_COLUMN, SPINE_TIMESTAMP_COL]


# ─── Generic: split, evaluate, champion, register ────────────────────────────


def _split(X, y, groups):
    """Grouped holdout so no entity appears in both train and test."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if TASK_TYPE == "regression":
        splitter = GroupKFold(n_splits=HOLDOUT_FOLDS)
        return next(splitter.split(X, y, groups))
    splitter = StratifiedGroupKFold(n_splits=HOLDOUT_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return next(splitter.split(X, y, groups))


def _predict_outputs(model, X):
    return getattr(model, PREDICT_FUNCTION)(X)


def _next_version(session) -> str:
    try:
        rows = session.sql(f"SHOW VERSIONS IN MODEL {DATABASE}.{SCHEMA}.{MODEL_NAME}").collect()
    except Exception:
        return "V1"
    nums = [int(r["name"][1:]) for r in rows if r["name"].startswith("V") and r["name"][1:].isdigit()]
    return f"V{max(nums) + 1}" if nums else "V1"


def score_champion(session, X_test, y_test) -> dict:
    """Score the champion (default version in champion.source_env) on the holdout.

    Returns {"available": False, "reason": ...} when there is no champion yet
    (first deployment) or it can't score this holdout (e.g. features changed).
    """
    if not CHAMPION_CONFIG.get("enabled"):
        return {"available": False, "reason": "champion comparison disabled"}

    from serving.signature import run_outputs
    from snowflake.ml.registry import Registry

    champ_db = env_database(CHAMPION_CONFIG.get("source_env", "prod"))
    try:
        reg = Registry(session=session, database_name=champ_db, schema_name=SCHEMA)
        mv = reg.get_model(MODEL_NAME).default
    except Exception as e:
        return {"available": False, "reason": f"no champion in {champ_db}.{SCHEMA}: {e}"}

    try:
        outputs = run_outputs(mv, X_test, PREDICT_FUNCTION)
    except Exception as e:
        return {"available": False, "reason": f"champion {mv.version_name} could not score holdout: {e}"}

    return {
        "available": True,
        "source": f"{champ_db}.{SCHEMA}.{MODEL_NAME}",
        "version": mv.version_name,
        "metrics": compute_metrics(TASK_TYPE, y_test, outputs),
    }


def _sample_input(dataset, feature_cols):
    """Snowpark sample derived from the Dataset, so the registry records lineage.

    Falls back to None (caller uses pandas) if the Dataset can't be read as a
    Snowpark DataFrame in this environment.
    """
    try:
        df = dataset.read.to_snowpark_dataframe().select(*feature_cols).limit(100)
        return df.na.fill(0)
    except Exception as e:
        print(f"  Lineage sample unavailable, using pandas sample: {e}")
        return None


def _registry_task():
    from snowflake.ml.model.task import Task

    return {
        "binary_classification": Task.TABULAR_BINARY_CLASSIFICATION,
        "multiclass": Task.TABULAR_MULTI_CLASSIFICATION,
        "regression": Task.TABULAR_REGRESSION,
    }[TASK_TYPE]


def register(session, model, version, metrics, sample, comment):
    """Log the model to the registry, via the experiment run when enabled."""
    kwargs = dict(
        model_name=MODEL_NAME,
        version_name=version,
        sample_input_data=sample,
        conda_dependencies=MODEL_DEPS,
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        metrics=metrics,
        comment=comment,
        # Required for ML Observability to pick the right monitoring metrics;
        # not inferable for custom models.
        task=_registry_task(),
    )
    if str(EXPERIMENT_CONFIG.get("enabled", "false")).lower() == "true":
        from datetime import datetime

        from snowflake.ml.experiment import ExperimentTracking

        exp = ExperimentTracking(session=session, database_name=DATABASE, schema_name=SCHEMA)
        exp.set_experiment(EXPERIMENT_CONFIG.get("experiment_name", f"{MODEL_NAME}_TRAINING"))
        run_name = f"{EXPERIMENT_CONFIG.get('run_name_prefix', 'pipeline')}_{version}_{datetime.now():%Y%m%d_%H%M%S}"
        with exp.start_run(run_name):
            exp.log_params({**{k: str(v) for k, v in TRAINING_PARAMS.items()}, "feature_view": FEATURE_VIEW})
            exp.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            return exp.log_model(model, **kwargs)

    from snowflake.ml.registry import Registry

    return Registry(session=session, database_name=DATABASE, schema_name=SCHEMA).log_model(model, **kwargs)


def run(session) -> dict:
    """Train, evaluate, score the champion, and register the candidate."""
    import json
    import os

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    dataset, df = build_dataset(session)
    feature_cols = [c for c in df.columns if c not in non_feature_columns()]
    X = df[feature_cols].fillna(0)
    y = df[LABEL_COLUMN]
    if TASK_TYPE != "regression":
        y = y.astype(int)
    groups = df[ID_COLUMNS[0]] if ID_COLUMNS else None

    train_idx, test_idx = _split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"  Dataset rows: {len(df):,} | features: {len(feature_cols)} | holdout: {len(X_test):,}")

    model = train_model(X_train, y_train)
    metrics = compute_metrics(TASK_TYPE, y_test, _predict_outputs(model, X_test))
    print(f"  Candidate metrics: {metrics}")

    champion = score_champion(session, X_test, y_test)
    print(f"  Champion: {champion.get('version', champion.get('reason'))}")

    version = _next_version(session)
    sample = _sample_input(dataset, feature_cols)
    if sample is None:
        sample = X_test.head(100)
    git_sha = os.getenv("GITHUB_SHA", "local")[:7]
    mv = register(
        session,
        model,
        version,
        metrics,
        sample,
        comment=f"features:{FEATURE_VIEW} | dataset:{DATASET_NAME}/{dataset.selected_version.name} | git:{git_sha}",
    )
    print(f"  Registered candidate: {DATABASE}.{SCHEMA}.{MODEL_NAME}/{mv.version_name}")

    result = {
        "status": "success",
        "step": "training",
        "model_version": mv.version_name,
        "task_type": TASK_TYPE,
        "metrics": metrics,
        "champion": champion,
        "feature_view": FEATURE_VIEW,
        "dataset": f"{DATASET_NAME}/{dataset.selected_version.name}",
        "dataset_rows": len(df),
    }
    session.sql(f"""
        INSERT INTO {DATABASE}.{SCHEMA}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
        SELECT 'training', 'SUCCESS', PARSE_JSON($${json.dumps(result)}$$), CURRENT_TIMESTAMP()
    """).collect()
    return result
