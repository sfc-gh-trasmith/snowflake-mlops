"""Generate synthetic fraud detection dataset and upload to Snowflake.

Produces three tables:
  - CUSTOMER_PROFILES: ~5,000 customers with demographics
  - MERCHANT_DATA: ~500 merchants with risk categories
  - RAW_TRANSACTIONS: ~100,000 transactions with ~3% fraud rate

Fraud patterns injected:
  - High amounts relative to customer history
  - Late-night / early-morning transactions
  - High-risk merchant categories
  - Velocity spikes (many txns in short window)
  - Geographic anomalies (large distance jumps)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import (
    DATABASE,
    SCHEMA,
    WAREHOUSE,
    CUSTOMER_PROFILES_TABLE,
    MERCHANT_DATA_TABLE,
    RAW_TRANSACTIONS_TABLE,
)
from snowpark_session import create_snowpark_session


def generate_customers(n: int = 5000, rng: np.random.Generator = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CUSTOMER_ID": [f"CUST_{i:05d}" for i in range(n)],
            "AGE": rng.integers(18, 75, size=n),
            "ANNUAL_INCOME": rng.lognormal(mean=10.8, sigma=0.5, size=n).astype(int),
            "ACCOUNT_AGE_DAYS": rng.integers(30, 3650, size=n),
            "CREDIT_SCORE": np.clip(rng.normal(700, 80, size=n), 300, 850).astype(int),
            "NUM_CARDS": rng.integers(1, 6, size=n),
            "STATE": rng.choice(
                [
                    "CA",
                    "TX",
                    "NY",
                    "FL",
                    "IL",
                    "PA",
                    "OH",
                    "GA",
                    "NC",
                    "MI",
                    "WA",
                    "AZ",
                    "MA",
                    "CO",
                    "VA",
                    "NJ",
                    "OR",
                    "TN",
                    "IN",
                    "MO",
                ],
                size=n,
            ),
        }
    )


def generate_merchants(n: int = 500, rng: np.random.Generator = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(43)
    categories = [
        "grocery",
        "gas_station",
        "restaurant",
        "online_retail",
        "electronics",
        "travel",
        "entertainment",
        "healthcare",
        "cash_advance",
        "gambling",
        "crypto_exchange",
        "jewelry",
    ]
    risk_map = {
        "grocery": 0.1,
        "gas_station": 0.15,
        "restaurant": 0.12,
        "online_retail": 0.3,
        "electronics": 0.35,
        "travel": 0.25,
        "entertainment": 0.2,
        "healthcare": 0.08,
        "cash_advance": 0.7,
        "gambling": 0.8,
        "crypto_exchange": 0.85,
        "jewelry": 0.5,
    }
    cats = rng.choice(categories, size=n)
    return pd.DataFrame(
        {
            "MERCHANT_ID": [f"MERCH_{i:04d}" for i in range(n)],
            "MERCHANT_NAME": [f"Merchant_{i}" for i in range(n)],
            "CATEGORY": cats,
            "RISK_SCORE": [risk_map[c] + rng.normal(0, 0.05) for c in cats],
            "AVG_TXN_AMOUNT": rng.lognormal(mean=3.5, sigma=1.0, size=n).round(2),
            "CITY": rng.choice(
                [
                    "New York",
                    "Los Angeles",
                    "Chicago",
                    "Houston",
                    "Phoenix",
                    "San Francisco",
                    "Seattle",
                    "Miami",
                    "Denver",
                    "Atlanta",
                ],
                size=n,
            ),
        }
    )


def generate_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n: int = 100_000,
    fraud_rate: float = 0.03,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(44)

    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    customer_ids = customers["CUSTOMER_ID"].values
    merchant_ids = merchants["MERCHANT_ID"].values

    # Base timestamp: 90 days of data
    base_time = pd.Timestamp("2024-01-01")
    end_time = pd.Timestamp("2024-03-31")
    time_range_seconds = int((end_time - base_time).total_seconds())

    # --- Legitimate transactions ---
    legit_customers = rng.choice(customer_ids, size=n_legit)
    legit_merchants = rng.choice(merchant_ids, size=n_legit)
    legit_amounts = rng.lognormal(mean=3.2, sigma=1.0, size=n_legit).round(2)
    legit_amounts = np.clip(legit_amounts, 1.0, 5000.0)
    legit_times = base_time + pd.to_timedelta(rng.integers(0, time_range_seconds, size=n_legit), unit="s")
    legit_hours = rng.choice(range(6, 23), size=n_legit, p=_daytime_probs())

    # --- Fraudulent transactions (with injected patterns) ---
    fraud_customers = rng.choice(customer_ids, size=n_fraud)
    fraud_merchants = rng.choice(merchants[merchants["RISK_SCORE"] > 0.4]["MERCHANT_ID"].values, size=n_fraud)
    # Higher amounts for fraud
    fraud_amounts = rng.lognormal(mean=5.5, sigma=1.2, size=n_fraud).round(2)
    fraud_amounts = np.clip(fraud_amounts, 50.0, 25000.0)
    fraud_times = base_time + pd.to_timedelta(rng.integers(0, time_range_seconds, size=n_fraud), unit="s")
    # Late-night hours for fraud
    fraud_hours = rng.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud)

    # Combine
    all_customers = np.concatenate([legit_customers, fraud_customers])
    all_merchants = np.concatenate([legit_merchants, fraud_merchants])
    all_amounts = np.concatenate([legit_amounts, fraud_amounts])
    all_times = np.concatenate([legit_times, fraud_times])
    all_hours = np.concatenate([legit_hours, fraud_hours])
    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    # Adjust timestamps to reflect actual hour
    all_times = pd.to_datetime(all_times)
    all_times = all_times.map(lambda t: t.replace(hour=0, minute=0, second=0)) + pd.to_timedelta(
        all_hours * 3600 + rng.integers(0, 3600, size=n), unit="s"
    )

    # Device and location
    devices = rng.choice(["mobile", "desktop", "tablet", "pos_terminal"], size=n, p=[0.45, 0.25, 0.1, 0.2])
    lats = rng.uniform(25.0, 48.0, size=n).round(4)
    lons = rng.uniform(-125.0, -70.0, size=n).round(4)

    # Shuffle
    idx = rng.permutation(n)

    df = pd.DataFrame(
        {
            "TXN_ID": [f"TXN_{i:07d}" for i in range(n)],
            "CUSTOMER_ID": all_customers[idx],
            "MERCHANT_ID": all_merchants[idx],
            "AMOUNT": all_amounts[idx],
            "TIMESTAMP": all_times[idx],
            "IS_FRAUD": is_fraud[idx],
            "DEVICE_TYPE": devices[idx],
            "LOCATION_LAT": lats[idx],
            "LOCATION_LON": lons[idx],
        }
    )

    return df.sort_values("TIMESTAMP").reset_index(drop=True)


def _daytime_probs():
    """Probability distribution for legitimate transaction hours (6-22)."""
    weights = [1, 2, 4, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(weights)
    return [w / total for w in weights]


def upload_to_snowflake(session, df: pd.DataFrame, table_name: str):
    """Write pandas DataFrame to Snowflake table."""
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    # Convert datetime columns to string for proper TIMESTAMP handling
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime64"]).columns:
        df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    tbl_short = table_name.split(".")[-1]
    session.write_pandas(df_copy, tbl_short, database=DATABASE, schema=SCHEMA, auto_create_table=True, overwrite=True)

    # Fix TIMESTAMP columns (write_pandas stores datetime-as-string as VARCHAR)
    if "TIMESTAMP" in df.columns:
        session.sql(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * EXCLUDE (TIMESTAMP),
                   TO_TIMESTAMP_NTZ(TIMESTAMP) AS TIMESTAMP
            FROM {table_name}
        """).collect()

    count = session.table(table_name).count()
    print(f"  {table_name}: {count:,} rows")


def main():
    print("Generating synthetic fraud detection dataset...")
    rng = np.random.default_rng(42)

    customers = generate_customers(n=5000, rng=rng)
    print(f"  Customers: {len(customers):,}")

    merchants = generate_merchants(n=500, rng=rng)
    print(f"  Merchants: {len(merchants):,}")

    transactions = generate_transactions(customers, merchants, n=100_000, rng=rng)
    print(f"  Transactions: {len(transactions):,} ({transactions['IS_FRAUD'].mean():.1%} fraud)")

    print("\nConnecting to Snowflake...")
    session = create_snowpark_session()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()

    print("Uploading tables...")
    upload_to_snowflake(session, customers, CUSTOMER_PROFILES_TABLE)
    upload_to_snowflake(session, merchants, MERCHANT_DATA_TABLE)
    upload_to_snowflake(session, transactions, RAW_TRANSACTIONS_TABLE)

    print("\nDataset generation complete!")
    session.close()


if __name__ == "__main__":
    main()
