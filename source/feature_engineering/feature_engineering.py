"""Feature engineering step.

Registers Feature Views in the Snowflake Feature Store. The feature logic
lives in feature_views.py alongside this module.

Runs identically on a warehouse (stored procedure) or on an SPCS compute pool
(ML Job); the caller decides which via pipeline_steps.feature_eng.compute.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION
from feature_engineering.data_checks import run_data_checks
from feature_engineering.feature_views import register_feature_views


def run(session) -> dict:
    """Validate source data, then register all Feature Views.

    Returns:
        dict with status, step name, data-check stats, and the registered
        feature view table name.
    """
    data_checks = run_data_checks(session)
    fs, customer_fv, txn_fv = register_feature_views(session=session)

    # NOTE: this string is consumed by source/pipeline/batch_inference_dag.py as the
    # batch scoring input table. Keep the NAME$VERSION shape.
    feature_view = f"{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}"

    return {
        "status": "success",
        "step": "feature_engineering",
        "data_checks": {t: {k: str(v) for k, v in s.items()} for t, s in data_checks.items()},
        "feature_view": feature_view,
        "feature_views_registered": [
            f"CUSTOMER_RISK_FEATURES${FEATURE_VIEW_VERSION}",
            f"TRANSACTION_CONTEXT_FEATURES${FEATURE_VIEW_VERSION}",
        ],
    }
