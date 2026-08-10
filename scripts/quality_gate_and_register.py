"""Quality Gate + Model Registration.

Reads metrics from the training pipeline, applies quality gate thresholds,
and if passed: registers the model artifact and replicates to PROD.

Exit code 1 if quality gate fails (model stays unregistered).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import MIN_AUC_ROC, MIN_PRECISION, MIN_RECALL, PIPELINE_CONFIG
from snowpark_session import create_snowpark_session

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOW_MLOPS_STAGE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", f"{DATABASE}_WH")
MODEL_NAME = PIPELINE_CONFIG.get("model_name", "MLOPS_FRAUD_DETECTOR")
PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"


def check_quality_gate(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    if metrics.get("auc_roc", 0) < MIN_AUC_ROC:
        failures.append(f"AUC-ROC {metrics['auc_roc']:.4f} < {MIN_AUC_ROC}")
    if metrics.get("precision", 0) < MIN_PRECISION:
        failures.append(f"Precision {metrics['precision']:.4f} < {MIN_PRECISION}")
    if metrics.get("recall", 0) < MIN_RECALL:
        failures.append(f"Recall {metrics['recall']:.4f} < {MIN_RECALL}")
    return len(failures) == 0, failures


def register_model(session, metrics: dict):
    """Register the trained model artifact from stage into the Model Registry."""
    import pandas as pd
    import xgboost as xgb
    from snowflake.ml.registry import Registry

    # Auto-increment version
    try:
        versions_df = session.sql(f"SHOW VERSIONS IN MODEL {DATABASE}.{SCHEMA}.{MODEL_NAME}").collect()
        existing = [r["name"] for r in versions_df]
        max_v = max(int(v.replace("V", "")) for v in existing if v.startswith("V") and v[1:].isdigit())
        version_name = f"V{max_v + 1}"
    except Exception:
        version_name = "V1"

    print(f"  Registering as: {DATABASE}.{SCHEMA}.{MODEL_NAME}/{version_name}")

    # Download model artifact from stage
    session.sql(f"GET @{DATABASE}.{SCHEMA}.PIPELINE_STAGE/artifacts/model.ubj file:///tmp/").collect()
    model = xgb.XGBClassifier()
    model.load_model("/tmp/model.ubj")

    # Download sample input
    session.sql(f"GET @{DATABASE}.{SCHEMA}.PIPELINE_STAGE/artifacts/sample_input.json file:///tmp/").collect()
    sample_input = pd.read_json("/tmp/sample_input.json", orient="records")

    # Register
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    fv = metrics.get("feature_view", "CUSTOMER_RISK_FEATURES$V1")
    git_sha = os.getenv("GITHUB_SHA", "unknown")[:7]

    reg.log_model(
        model=model,
        model_name=MODEL_NAME,
        version_name=version_name,
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=sample_input,
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        comment=f"features:{fv} | AUC={metrics.get('auc_roc', 0):.4f} | git:{git_sha}",
    )

    return version_name


def replicate_to_prod(session, version: str):
    """Copy model version from STAGE to PROD."""
    import importlib

    topology = os.getenv("TOPOLOGY", "single-account")
    strategy_map = {
        "single-account": "deploy.strategies.single_account",
        "multi-account": "deploy.strategies.multi_account",
        "cross-region": "deploy.strategies.cross_region",
    }

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_path = strategy_map.get(topology, strategy_map["single-account"])
    module = importlib.import_module(module_path)
    module.promote(version=version, session=session)


def write_summary(metrics: dict, version: str, passed: bool):
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    status = "PASSED" if passed else "FAILED"
    with open(summary_path, "a") as f:
        f.write(f"\n## Quality Gate: {status}\n\n")
        f.write(f"**Model**: `{DATABASE}.{SCHEMA}.{MODEL_NAME}/{version}`\n\n")
        f.write("| Metric | Value | Threshold | Status |\n|--------|-------|-----------|--------|\n")
        f.write(
            f"| AUC-ROC | {metrics.get('auc_roc', 0):.4f} | >= {MIN_AUC_ROC} | {'PASS' if metrics.get('auc_roc', 0) >= MIN_AUC_ROC else 'FAIL'} |\n"
        )
        f.write(
            f"| Precision | {metrics.get('precision', 0):.4f} | >= {MIN_PRECISION} | {'PASS' if metrics.get('precision', 0) >= MIN_PRECISION else 'FAIL'} |\n"
        )
        f.write(
            f"| Recall | {metrics.get('recall', 0):.4f} | >= {MIN_RECALL} | {'PASS' if metrics.get('recall', 0) >= MIN_RECALL else 'FAIL'} |\n"
        )


def main():
    print("=" * 60)
    print("QUALITY GATE + MODEL REGISTRATION")
    print("=" * 60)

    # Load metrics from previous step
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found: {metrics_file}")
        sys.exit(1)

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Quality gate
    print("\n[1/3] Quality gate check...")
    passed, failures = check_quality_gate(metrics)

    if passed:
        print("  PASSED — all metrics meet thresholds")
    else:
        print("  FAILED:")
        for failure in failures:
            print(f"    - {failure}")
        write_summary(metrics, "N/A", False)
        print("\nModel will NOT be registered. Fix and re-run.")
        sys.exit(1)

    # Register model
    print("\n[2/3] Registering model...")
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    version = register_model(session, metrics)
    print(f"  Registered: {MODEL_NAME}/{version}")

    # Replicate to PROD
    print(f"\n[3/3] Promoting model {version} to PROD...")
    replicate_to_prod(session, version)
    print(f"  Model replicated to {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}/{version}")

    write_summary(metrics, version, True)

    print("\n" + "=" * 60)
    print(f"MODEL PROMOTED: {MODEL_NAME}/{version}")
    print("  Quality gate: PASSED")
    print(f"  Registered in: {DATABASE}.{SCHEMA}")
    print(f"  Replicated to: {PROD_DATABASE}.{PROD_SCHEMA}")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
