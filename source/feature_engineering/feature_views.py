"""Define and register Feature Views for fraud detection.

Feature Views:
  1. CUSTOMER_RISK_FEATURES - Aggregated customer behavior signals
  2. TRANSACTION_CONTEXT_FEATURES - Per-transaction contextual signals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import snowflake.snowpark.functions as F
from config import (
    DATABASE,
    FEATURE_VIEW_VERSION,
    SCHEMA,
    SOURCE_DATABASE,
    SOURCE_SCHEMA,
    WAREHOUSE,
)

# Feature View refresh intervals (from config.yml)
from config import _cfg as _raw_cfg
from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
from snowflake.snowpark import Session

_fs_cfg = _raw_cfg.get("feature_store", {})
CUSTOMER_FEATURES_REFRESH = _fs_cfg.get("customer_features_refresh", "1 hour")
TRANSACTION_FEATURES_REFRESH = _fs_cfg.get("transaction_features_refresh", "1 hour")


def create_customer_features_df(session: Session):
    """Build Snowpark DataFrame for customer-level risk features."""
    txn = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS")
    cust = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.CUSTOMER_PROFILES")

    customer_agg = txn.group_by("CUSTOMER_ID").agg(
        F.count("*").alias("TOTAL_TXN_COUNT"),
        F.avg("AMOUNT").alias("AVG_TXN_AMOUNT"),
        F.max("AMOUNT").alias("MAX_TXN_AMOUNT"),
        F.stddev("AMOUNT").alias("STDDEV_TXN_AMOUNT"),
        F.count_distinct("MERCHANT_ID").alias("UNIQUE_MERCHANTS"),
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
        F.col("ACTIVE_DAYS"),
        F.col("LATE_NIGHT_TXN_RATIO"),
        F.col("CREDIT_SCORE"),
        F.col("ACCOUNT_AGE_DAYS"),
        F.col("ANNUAL_INCOME"),
        F.col("FEATURE_TS"),
    )
    return features


def create_transaction_features_df(session: Session):
    """Build Snowpark DataFrame for transaction-level context features."""
    txn = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS")
    merch = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.MERCHANT_DATA")
    cust_avg = txn.group_by("CUSTOMER_ID").agg(F.avg("AMOUNT").alias("CUST_AVG_AMOUNT"))

    features = (
        txn.join(merch, on="MERCHANT_ID", how="left")
        .join(cust_avg, on="CUSTOMER_ID", how="left")
        .select(
            F.col("TXN_ID"),
            F.col("CUSTOMER_ID"),
            F.col("AMOUNT"),
            (F.col("AMOUNT") / F.coalesce(F.col("CUST_AVG_AMOUNT"), F.lit(1.0))).alias("AMOUNT_TO_AVG_RATIO"),
            F.when(F.col("RISK_SCORE") > 0.5, F.lit(1)).otherwise(F.lit(0)).alias("IS_HIGH_RISK_MERCHANT"),
            F.col("RISK_SCORE").alias("MERCHANT_RISK_SCORE"),
            F.hour("TIMESTAMP").alias("HOUR_OF_DAY"),
            F.when(F.dayofweek("TIMESTAMP").isin([0, 6]), F.lit(1)).otherwise(F.lit(0)).alias("IS_WEEKEND"),
            F.when((F.hour("TIMESTAMP") < 6) | (F.hour("TIMESTAMP") > 22), F.lit(1))
            .otherwise(F.lit(0))
            .alias("IS_LATE_NIGHT"),
            F.col("DEVICE_TYPE"),
            F.col("CATEGORY").alias("MERCHANT_CATEGORY"),
            F.col("TIMESTAMP").alias("FEATURE_TS"),
        )
    )
    return features


def _schema(df) -> dict:
    return {f.name: str(f.datatype) for f in df.schema.fields}


def register_versioned(fs, feature_view: FeatureView, version: str) -> FeatureView:
    """Register a feature view version, treating versions as immutable.

    - New version: created (initial backfill runs).
    - Existing version with the same output schema: reused as-is (no rebuild).
    - Existing version whose feature columns changed: fails, because models
      trained on that version would silently start reading different features.
      Bump feature_store.view_version in config.yml instead.

    Logic-only changes that keep the same columns can't be detected reliably,
    so bump the version for any change to the feature SQL.
    """
    try:
        existing = fs.get_feature_view(feature_view.name, version)
    except Exception:
        existing = None

    if existing is not None:
        new_cols = _schema(feature_view.feature_df)
        old_cols = _schema(existing.feature_df)
        if new_cols != old_cols:
            added = sorted(set(new_cols) - set(old_cols))
            removed = sorted(set(old_cols) - set(new_cols))
            changed = sorted(c for c in set(new_cols) & set(old_cols) if new_cols[c] != old_cols[c])
            raise RuntimeError(
                f"Feature view {feature_view.name}/{version} already exists with a different schema "
                f"(added={added}, removed={removed}, type_changed={changed}). "
                "Bump feature_store.view_version in source/configs/config.yml to register a new version."
            )
        print(f"  Reusing existing {feature_view.name}/{version} (immutable version, schema unchanged)")
        return existing

    registered = fs.register_feature_view(feature_view=feature_view, version=version)
    print(f"  Registered new version {feature_view.name}/{version}")
    return registered


def register_feature_views(session=None, database=None, schema=None, warehouse=None):
    """Register all feature views in the Feature Store. Idempotent per version."""
    close_session = False
    if session is None:
        from snowpark_session import create_snowpark_session

        session = create_snowpark_session()
        close_session = True

    db = database or DATABASE
    sch = schema or SCHEMA
    wh = warehouse or WAREHOUSE

    session.sql(f"USE WAREHOUSE {wh}").collect()

    fs = FeatureStore(
        session=session,
        database=db,
        name=sch,
        default_warehouse=wh,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
    transaction_entity = Entity(name="TRANSACTION", join_keys=["TXN_ID"])

    fs.register_entity(customer_entity)
    fs.register_entity(transaction_entity)

    print("Creating CUSTOMER_RISK_FEATURES feature view...")
    cust_df = create_customer_features_df(session)
    customer_fv = FeatureView(
        name="CUSTOMER_RISK_FEATURES",
        entities=[customer_entity],
        feature_df=cust_df,
        timestamp_col="FEATURE_TS",
        refresh_freq=CUSTOMER_FEATURES_REFRESH,
        desc="Customer-level risk signals for fraud detection",
    )
    customer_fv = register_versioned(fs, customer_fv, FEATURE_VIEW_VERSION)
    print(f"  CUSTOMER_RISK_FEATURES/{FEATURE_VIEW_VERSION} (refresh={CUSTOMER_FEATURES_REFRESH})")

    print("Creating TRANSACTION_CONTEXT_FEATURES feature view...")
    txn_df = create_transaction_features_df(session)
    txn_fv = FeatureView(
        name="TRANSACTION_CONTEXT_FEATURES",
        entities=[transaction_entity],
        feature_df=txn_df,
        timestamp_col="FEATURE_TS",
        refresh_freq=TRANSACTION_FEATURES_REFRESH,
        desc="Per-transaction contextual signals for fraud detection",
    )
    txn_fv = register_versioned(fs, txn_fv, FEATURE_VIEW_VERSION)
    print(f"  TRANSACTION_CONTEXT_FEATURES/{FEATURE_VIEW_VERSION} (refresh={TRANSACTION_FEATURES_REFRESH})")

    if close_session:
        session.close()

    return fs, customer_fv, txn_fv


if __name__ == "__main__":
    register_feature_views()
