"""Set the PROD model default version (used by the rollback workflow).

Usage:
  python deploy/set_default_version.py --version V4
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import MODEL_NAME, SCHEMA, env_database
from snowpark_session import create_snowpark_session


def main():
    parser = argparse.ArgumentParser(description="Set the model default version")
    parser.add_argument("--version", required=True, help="Version to make default (e.g. V4)")
    args = parser.parse_args()

    version = args.version.strip().upper()
    if not (version.startswith("V") and version[1:].isdigit()):
        sys.exit(f"Invalid version '{args.version}' — expected V<number>")

    db = os.getenv("SNOWFLAKE_DATABASE", env_database("prod"))
    schema = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)
    model_fqn = f"{db}.{schema}.{MODEL_NAME}"

    session = create_snowpark_session()
    versions = {r["name"] for r in session.sql(f"SHOW VERSIONS IN MODEL {model_fqn}").collect()}
    if version not in versions:
        session.close()
        sys.exit(f"{version} not found in {model_fqn}. Available: {sorted(versions)}")

    session.sql(f"ALTER MODEL {model_fqn} SET DEFAULT_VERSION = {version}").collect()
    print(f"Default version of {model_fqn} set to {version}")
    session.close()


if __name__ == "__main__":
    main()
