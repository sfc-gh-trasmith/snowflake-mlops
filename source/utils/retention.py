"""Retention: prune old model versions and training Dataset versions.

Every pipeline run registers a model version and a Dataset version, so both
grow without bound. This keeps the newest N of each (retention.* in
config.yml) and never removes anything still in use:

  model versions  - the default version, any version with an inference
                    service, any version a model monitor watches
  dataset versions - any version referenced in the lineage comment of a
                    remaining model version in this environment or in PROD
                    (promoted models point at the STAGE dataset they came from)

Run by deploy-stage (stage) and deploy-prod (prod) after a successful deploy.
Failures are reported but never fail the deploy.

Usage:
  python source/utils/retention.py --env stage [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL_NAME, RETENTION_CONFIG, SCHEMA, env_database, resolve_env_config
from snowpark_session import create_snowpark_session

_DATASET_REF = re.compile(r"dataset:([^/\s|]+)/([^\s|]+)")


def _version_number(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else -1


def plan_model_versions(versions: list[dict], keep: int, protected: set[str]) -> list[str]:
    """Versions to drop: everything outside the newest `keep`, minus protected ones."""
    ordered = sorted(versions, key=lambda v: _version_number(v["name"]), reverse=True)
    newest = {v["name"] for v in ordered[:keep]}
    return [v["name"] for v in ordered if v["name"] not in newest and v["name"] not in protected]


def plan_dataset_versions(versions: list[str], keep: int, referenced: set[str]) -> list[str]:
    """Dataset versions to drop. Names are <FV version>_<UTC timestamp>, so sort by timestamp."""
    ordered = sorted(versions, key=lambda v: v.split("_", 1)[-1], reverse=True)
    return [v for v in ordered[keep:] if v not in referenced]


def dataset_refs(comments: list[str]) -> set[tuple[str, str]]:
    """(dataset, version) pairs from 'dataset:<NAME>/<VERSION>' lineage comments."""
    return {m for c in comments for m in _DATASET_REF.findall(c or "")}


def _show(session, sql: str) -> list:
    try:
        return session.sql(sql).collect()
    except Exception as e:
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            return []
        raise


def _model_versions(session, db: str, schema: str) -> list[dict]:
    rows = _show(session, f"SHOW VERSIONS IN MODEL {db}.{schema}.{MODEL_NAME}")
    return [
        {
            "name": r["name"],
            "comment": r["comment"] or "",
            "default": str(r["is_default_version"]).lower() == "true",
            "services": json.loads(r["inference_services"] or "[]"),
        }
        for r in rows
    ]


def _monitored_versions(session, db: str, schema: str) -> set[str]:
    versions = set()
    for r in _show(session, f"SHOW MODEL MONITORS IN SCHEMA {db}.{schema}"):
        model = json.loads(r["model"] or "{}")
        if model.get("model_name") == MODEL_NAME and model.get("version_name"):
            versions.add(model["version_name"])
    return versions


def prune(session, env: str, dry_run: bool) -> dict:
    cfg = resolve_env_config(env)
    db, schema = cfg["database"], cfg["schema"]
    keep_models = int(RETENTION_CONFIG.get("keep_model_versions", 5))
    keep_datasets = int(RETENTION_CONFIG.get("keep_dataset_versions", 5))
    verb = "Would drop" if dry_run else "Dropping"
    report = {"model_versions": [], "dataset_versions": [], "errors": []}

    versions = _model_versions(session, db, schema)
    protected = {v["name"] for v in versions if v["default"] or v["services"]} | _monitored_versions(
        session, db, schema
    )
    for name in plan_model_versions(versions, keep_models, protected):
        print(f"  {verb} model version {MODEL_NAME}/{name}")
        try:
            if not dry_run:
                session.sql(f"ALTER MODEL {db}.{schema}.{MODEL_NAME} DROP VERSION {name}").collect()
            report["model_versions"].append(name)
        except Exception as e:
            report["errors"].append(f"{name}: {e}")

    # Lineage from what's left here, plus PROD (its versions point at STAGE datasets).
    dropped = set(report["model_versions"])
    remaining = [v["comment"] for v in versions if v["name"] not in dropped]
    if env != "prod":
        remaining += [v["comment"] for v in _model_versions(session, env_database("prod"), SCHEMA)]
    refs = dataset_refs(remaining)

    for ds in _show(session, f"SHOW DATASETS IN SCHEMA {db}.{schema}"):
        name = ds["name"]
        ds_versions = json.loads(ds["versions"] or "[]")
        referenced = {v for d, v in refs if d == name}
        for version in plan_dataset_versions(ds_versions, keep_datasets, referenced):
            print(f"  {verb} dataset version {name}/{version}")
            try:
                if not dry_run:
                    session.sql(f"ALTER DATASET {db}.{schema}.{name} DROP VERSION '{version}'").collect()
                report["dataset_versions"].append(f"{name}/{version}")
            except Exception as e:
                report["errors"].append(f"{name}/{version}: {e}")

    print(
        f"  {env.upper()}: {len(report['model_versions'])} model version(s), "
        f"{len(report['dataset_versions'])} dataset version(s) {'to drop' if dry_run else 'dropped'}"
    )
    for err in report["errors"]:
        print(f"::warning::Retention: {err}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Prune old model and dataset versions")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "dev"), choices=["dev", "stage", "prod"])
    parser.add_argument("--dry-run", action="store_true", help="Print what would be dropped")
    args = parser.parse_args()

    if str(RETENTION_CONFIG.get("enabled", "true")).lower() != "true":
        print("Retention disabled (retention.enabled: false)")
        return

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {resolve_env_config(args.env)['warehouse']}").collect()
    report = prune(session, args.env, args.dry_run)
    session.close()

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary and (report["model_versions"] or report["dataset_versions"]):
        with open(summary, "a") as f:
            f.write(f"\n## Retention ({args.env.upper()})\n\n")
            if report["model_versions"]:
                f.write("Dropped model versions: " + ", ".join(f"`{v}`" for v in report["model_versions"]) + "\n\n")
            if report["dataset_versions"]:
                f.write("Dropped dataset versions: " + ", ".join(f"`{v}`" for v in report["dataset_versions"]) + "\n")


if __name__ == "__main__":
    main()
