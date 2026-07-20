"""Register trained model to Snowflake Model Registry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE, SCHEMA, WAREHOUSE, MODEL_NAME
from snowpark_session import create_snowpark_session

import pandas as pd


def register_model(
    model,
    sample_input: pd.DataFrame,
    version_name: str = "V1",
    session=None,
    comment: str = "XGBoost fraud detection classifier",
) -> object:
    """Log model to Snowflake Model Registry.

    Args:
        model: Trained model (XGBoost, sklearn, etc.)
        sample_input: Sample input DataFrame for schema inference
        version_name: Version label (e.g., "V1", "V2")
        session: Optional Snowpark session
        comment: Description for this model version

    Returns:
        ModelVersion object
    """
    from snowflake.ml.registry import Registry

    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)

    mv = reg.log_model(
        model=model,
        model_name=MODEL_NAME,
        version_name=version_name,
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=sample_input,
        comment=comment,
    )

    print(f"Model registered: {DATABASE}.{SCHEMA}.{MODEL_NAME}/{version_name}")
    print(f"  Methods: {mv.show_functions()}")

    if close_session:
        session.close()

    return mv


def get_model_version(version_name: str = "V1", session=None):
    """Retrieve a registered model version."""
    from snowflake.ml.registry import Registry

    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(version_name)

    if close_session:
        session.close()

    return mv


if __name__ == "__main__":
    print("Use register_model(model, sample_input) to register a trained model.")
