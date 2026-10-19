"""Promote a model version from STAGE to PROD.

Copies the version between registries in the same account:

    ALTER MODEL PROD_DB.SCHEMA.MODEL ADD VERSION V{n}
      FROM MODEL STAGE_DB.SCHEMA.MODEL VERSION V{n}

then makes it the PROD default. Promotion is idempotent: re-running it for a
version that's already in PROD (e.g. retrying a deploy that failed at a later
step) keeps the existing version, which may already back a live inference
service, instead of dropping and re-copying it.

Cross-account promotion is not part of this template; see docs/docs.html.

Usage:
  python deploy/promote.py --version V3
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from config import MODEL_NAME, SCHEMA, env_database
from snowpark_session import create_snowpark_session

# Source and target registries come from source/configs/config.yml. CI can
# override them with the GitHub variables passed as these env vars.
STAGE_DATABASE = os.getenv("SNOWFLAKE_DATABASE_STAGE", env_database("stage"))
PROD_DATABASE = os.getenv("SNOWFLAKE_DATABASE_PROD", env_database("prod"))
STAGE_SCHEMA = PROD_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)


def _version_row(session, fqn: str, version: str):
    """SHOW VERSIONS row for one version, or None if it (or the model) doesn't exist."""
    try:
        rows = session.sql(f"SHOW VERSIONS LIKE '{version}' IN MODEL {fqn}").collect()
    except Exception as e:
        if "does not exist" in str(e).lower():
            return None
        raise
    return rows[0] if rows else None


def same_version(prod_row, stage_row) -> bool:
    """True if a PROD version is the copy of this STAGE version.

    The STAGE comment carries the dataset version (timestamped per run) and git
    commit, so it identifies the training run. Copies made before comments were
    carried over have an empty comment; fall back to size + metadata for those.
    """
    if prod_row["comment"]:
        return prod_row["comment"] == stage_row["comment"]
    return prod_row["size"] == stage_row["size"] and prod_row["metadata"] == stage_row["metadata"]


def _sql_string(value: str) -> str:
    return "'" + (value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"


def promote(version: str, session=None):
    """Replicate a model version from STAGE to PROD (cross-database copy)."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    stage_fqn = f"{STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}"
    prod_fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}"
    print(f"Promoting {MODEL_NAME}/{version}: {STAGE_DATABASE} -> {PROD_DATABASE}")

    stage_row = _version_row(session, stage_fqn, version)
    if stage_row is None:
        raise RuntimeError(f"{stage_fqn} has no version {version} to promote")

    prod_row = _version_row(session, prod_fqn, version)
    if prod_row is not None:
        if not same_version(prod_row, stage_row):
            raise RuntimeError(
                f"{prod_fqn} already has a different {version} (PROD comment: {prod_row['comment']!r}, "
                f"STAGE comment: {stage_row['comment']!r}). Version names must be unique across "
                "promotions; this happens if STAGE was rebuilt and restarted numbering. Retrain to get "
                "a new version name, or remove the PROD version deliberately if it isn't serving."
            )
        print(f"  {version} is already in PROD (same training run) - keeping it")
    else:
        model_exists = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
        if model_exists:
            session.sql(
                f"ALTER MODEL {prod_fqn} ADD VERSION {version} FROM MODEL {stage_fqn} VERSION {version}"
            ).collect()
        else:
            session.sql(
                f"CREATE MODEL {prod_fqn} WITH VERSION {version} FROM MODEL {stage_fqn} VERSION {version}"
            ).collect()
        print(f"  Copied {version} to PROD")

    # The copy keeps metrics but not the comment; carry the lineage over
    # (features, dataset version, git commit).
    if stage_row["comment"]:
        session.sql(
            f"ALTER MODEL {prod_fqn} MODIFY VERSION {version} SET COMMENT = {_sql_string(stage_row['comment'])}"
        ).collect()

    session.sql(f"ALTER MODEL {prod_fqn} SET DEFAULT_VERSION = {version}").collect()

    result = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    if not result:
        raise RuntimeError("Promotion failed - model not found in PROD after replication")
    print(f"  Promoted successfully! Default version: {version}")
    print(f"  All PROD versions: {result[0]['versions']}")

    if close_session:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Promote a model version from STAGE to PROD")
    parser.add_argument("--version", required=True, help="Model version to promote (e.g., V3)")
    args = parser.parse_args()
    promote(version=args.version)
    print("\nPromotion complete.")


if __name__ == "__main__":
    main()
