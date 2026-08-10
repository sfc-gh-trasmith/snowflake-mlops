"""Feature Engineering task — registers Feature Views in the target environment.

Runs as a Snowflake ML Job (warehouse or SPCS depending on config).
Called by the ML_PIPELINE_FEATURE_ENG task in the Task DAG.

Writes output to PIPELINE_RESULTS table for downstream tasks.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from features.feature_views import register_feature_views
from snowflake.snowpark import Session


def main():
    session = Session.builder.getOrCreate()

    database = os.getenv("PIPELINE_DATABASE", "SNOW_MLOPS_STAGE")
    schema = os.getenv("PIPELINE_SCHEMA", "ML")
    warehouse = os.getenv("PIPELINE_WAREHOUSE", f"{database}_WH")

    print(f"Feature Engineering: registering views in {database}.{schema}")
    session.sql(f"USE WAREHOUSE {warehouse}").collect()

    register_feature_views(session=session, database=database, schema=schema, warehouse=warehouse)

    result = {"status": "success", "step": "feature_engineering", "database": database}
    session.sql(f"""
        INSERT INTO {database}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
        VALUES ('feature_engineering', 'SUCCESS', PARSE_JSON('{json.dumps(result)}'), CURRENT_TIMESTAMP())
    """).collect()

    print("Feature engineering complete.")
    session.close()


if __name__ == "__main__":
    main()
