"""Setup Feature Store in STAGE environment.

Creates the same Feature Views (Dynamic Tables) as DEV but in SNOW_MLOPS_STAGE.ML.
Source data comes from SNOW_MLOPS_PROD.ML (always).
Run this once before the first STAGE pipeline execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, FeatureView, Entity, CreationMode
import snowflake.snowpark.functions as F

STAGE_DATABASE = "SNOW_MLOPS_STAGE"
STAGE_SCHEMA = "ML"
STAGE_WAREHOUSE = "SNOW_MLOPS_STAGE_WH"

SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"


def setup_stage_feature_store():
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {STAGE_WAREHOUSE}").collect()

    fs = FeatureStore(
        session=session,
        database=STAGE_DATABASE,
        name=STAGE_SCHEMA,
        default_warehouse=STAGE_WAREHOUSE,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    # Register entities
    customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
    fs.register_entity(customer_entity)
    print("Registered entity: CUSTOMER")

    # Customer risk features (reads from PROD source data)
    txn = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS")
    cust = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.CUSTOMER_PROFILES")

    customer_agg = txn.group_by("CUSTOMER_ID").agg(
        F.count("*").alias("TOTAL_TXN_COUNT"),
        F.avg("AMOUNT").alias("AVG_TXN_AMOUNT"),
        F.max("AMOUNT").alias("MAX_TXN_AMOUNT"),
        F.stddev("AMOUNT").alias("STDDEV_TXN_AMOUNT"),
        F.count_distinct("MERCHANT_ID").alias("UNIQUE_MERCHANTS"),
        F.sum(F.when(F.col("IS_FRAUD") == 1, F.lit(1)).otherwise(F.lit(0))).alias("HISTORICAL_FRAUD_COUNT"),
        F.avg(F.col("IS_FRAUD").cast("float")).alias("HISTORICAL_FRAUD_RATE"),
        F.count_distinct(F.dayofyear("TIMESTAMP")).alias("ACTIVE_DAYS"),
        F.avg(
            F.when(F.hour("TIMESTAMP") < 6, F.lit(1)).when(F.hour("TIMESTAMP") > 22, F.lit(1)).otherwise(F.lit(0))
        ).alias("LATE_NIGHT_TXN_RATIO"),
        F.max("TIMESTAMP").alias("FEATURE_TS"),
    )

    features = customer_agg.join(cust, on="CUSTOMER_ID", how="inner").select(
        F.col("CUSTOMER_ID"),
        F.col("TOTAL_TXN_COUNT"),
        F.col("AVG_TXN_AMOUNT"),
        F.col("MAX_TXN_AMOUNT"),
        F.col("STDDEV_TXN_AMOUNT"),
        F.col("UNIQUE_MERCHANTS"),
        F.col("HISTORICAL_FRAUD_COUNT"),
        F.col("HISTORICAL_FRAUD_RATE"),
        F.col("ACTIVE_DAYS"),
        F.col("LATE_NIGHT_TXN_RATIO"),
        F.col("CREDIT_SCORE"),
        F.col("ACCOUNT_AGE_DAYS"),
        F.col("ANNUAL_INCOME"),
        F.col("FEATURE_TS"),
    )

    customer_fv = FeatureView(
        name="CUSTOMER_RISK_FEATURES",
        entities=[customer_entity],
        feature_df=features,
        timestamp_col="FEATURE_TS",
        refresh_freq="1 hour",
        desc="Customer-level risk signals (STAGE, reads from PROD source data)",
    )
    fs.register_feature_view(feature_view=customer_fv, version="V1", overwrite=True)
    print("Registered: CUSTOMER_RISK_FEATURES/V1 in STAGE")

    session.close()
    print("\nSTAGE Feature Store setup complete!")


if __name__ == "__main__":
    setup_stage_feature_store()
