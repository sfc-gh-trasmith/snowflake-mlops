"""Pre-training data checks.

Fails the pipeline early when upstream data looks wrong, before features or
models are built on it. Rules come from data_checks in config.yml and run as
single aggregate queries per table in the source database/schema.

  min_rows          minimum row count
  max_null_pct      {column: max % NULLs (0-100)}
  freshness_column  timestamp column; with max_age_hours, the newest row must
                    be at most that many hours old
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_CHECKS, SOURCE_DATABASE, SOURCE_SCHEMA


def build_check_sql(table_fqn: str, rules: dict) -> str:
    """One aggregate query returning every metric the rules need."""
    selects = ["COUNT(*) AS ROW_COUNT"]
    for i, col in enumerate((rules.get("max_null_pct") or {}).keys()):
        selects.append(f"100 * COUNT_IF({col} IS NULL) / NULLIF(COUNT(*), 0) AS NULL_PCT_{i}")
    if rules.get("freshness_column") and rules.get("max_age_hours") is not None:
        selects.append(f"DATEDIFF('hour', MAX({rules['freshness_column']}), CURRENT_TIMESTAMP()) AS AGE_HOURS")
    return f"SELECT {', '.join(selects)} FROM {table_fqn}"


def evaluate_rules(table: str, rules: dict, stats: dict) -> list[str]:
    """Compare query results to rules. Returns failures (empty = passed)."""
    failures = []
    rows = stats.get("ROW_COUNT") or 0
    if rules.get("min_rows") is not None and rows < int(rules["min_rows"]):
        failures.append(f"{table}: {rows:,} rows < min_rows {int(rules['min_rows']):,}")
    for i, (col, max_pct) in enumerate((rules.get("max_null_pct") or {}).items()):
        pct = float(stats.get(f"NULL_PCT_{i}") or 0.0)
        if pct > float(max_pct):
            failures.append(f"{table}.{col}: {pct:.2f}% NULL > max {max_pct}%")
    if "AGE_HOURS" in stats and stats["AGE_HOURS"] is not None:
        if stats["AGE_HOURS"] > float(rules["max_age_hours"]):
            failures.append(
                f"{table}: newest {rules['freshness_column']} is {stats['AGE_HOURS']}h old "
                f"> max_age_hours {rules['max_age_hours']}"
            )
    return failures


def run_data_checks(session, checks: dict | None = None) -> dict:
    """Run all configured checks. Raises RuntimeError listing every failure."""
    checks = DATA_CHECKS if checks is None else checks
    failures, report = [], {}
    for table, rules in checks.items():
        table_fqn = table if "." in table else f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.{table}"
        try:
            stats = session.sql(build_check_sql(table_fqn, rules or {})).collect()[0].as_dict()
        except Exception as e:
            failures.append(f"{table}: check query failed ({e})")
            continue
        report[table] = stats
        failures.extend(evaluate_rules(table, rules or {}, stats))

    for table, stats in report.items():
        print(f"  Data check {table}: {stats}")
    if failures:
        raise RuntimeError("Data checks failed:\n  - " + "\n  - ".join(failures))
    return report
