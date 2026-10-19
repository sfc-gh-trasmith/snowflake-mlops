"""Model evaluation step: the quality gate.

Reads the candidate registered by the training step, checks it against the
absolute metric bars and the champion, and records the decision in
PIPELINE_RESULTS (STEP='evaluation').

  passed -> candidate stays registered, tagged gate=PASSED via set_metric
  failed -> candidate version is removed from the registry and the step
            raises, so the DAG (and CI) fails and nothing can be promoted
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHAMPION_CONFIG, DATABASE, GATE_RULES, MODEL_NAME, SCHEMA
from evaluate.gate import check_rules, compare_champion


def _remove_candidate(session, version: str):
    """Drop a rejected candidate. The very first version is the model's
    default and can't be dropped on its own, so drop the model instead."""
    model_fqn = f"{DATABASE}.{SCHEMA}.{MODEL_NAME}"
    versions = session.sql(f"SHOW VERSIONS IN MODEL {model_fqn}").collect()
    if len(versions) <= 1:
        session.sql(f"DROP MODEL IF EXISTS {model_fqn}").collect()
        print(f"  Dropped model {model_fqn} (rejected candidate was its only version)")
    else:
        session.sql(f"ALTER MODEL {model_fqn} DROP VERSION {version}").collect()
        print(f"  Dropped rejected candidate {model_fqn}/{version}")


def run(session) -> dict:
    """Apply the quality gate to the latest training candidate."""
    import json

    rows = session.sql(f"""
        SELECT RESULT FROM {DATABASE}.{SCHEMA}.PIPELINE_RESULTS
        WHERE STEP = 'training' AND STATUS = 'SUCCESS'
        ORDER BY CREATED_AT DESC LIMIT 1
    """).collect()
    if not rows:
        raise RuntimeError("EVALUATE failed: no training results found in PIPELINE_RESULTS")

    training = json.loads(rows[0]["RESULT"])
    version = training["model_version"]
    metrics = training.get("metrics", {})
    champion = training.get("champion", {})

    failures = check_rules(metrics, GATE_RULES)
    champ_passed, champ_note = compare_champion(metrics, champion, CHAMPION_CONFIG)
    if not champ_passed:
        failures.append(champ_note)
    passed = not failures

    print(f"  Candidate: {MODEL_NAME}/{version}")
    print(f"  Champion: {champ_note}")
    print(f"  Gate: {'PASSED' if passed else 'FAILED'}")
    for f in failures:
        print(f"    - {f}")

    result = {
        "status": "success" if passed else "failed",
        "step": "evaluation",
        "model_version": version,
        "task_type": training.get("task_type"),
        "metrics": metrics,
        "champion": champion,
        "champion_comparison": champ_note,
        "gate_rules": GATE_RULES,
        "gate_passed": passed,
        "gate_failures": failures,
        "pipeline_status": "READY_FOR_REVIEW" if passed else "REJECTED",
        # Lineage, surfaced in the CI job summary
        "feature_view": training.get("feature_view"),
        "dataset": training.get("dataset"),
        "dataset_rows": training.get("dataset_rows"),
        "git_sha": os.getenv("GITHUB_SHA", "local")[:7],
    }
    session.sql(f"""
        INSERT INTO {DATABASE}.{SCHEMA}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
        SELECT 'evaluation', '{"SUCCESS" if passed else "FAILED"}', PARSE_JSON($${json.dumps(result)}$$),
               CURRENT_TIMESTAMP()
    """).collect()

    if passed:
        from snowflake.ml.registry import Registry

        mv = (
            Registry(session=session, database_name=DATABASE, schema_name=SCHEMA).get_model(MODEL_NAME).version(version)
        )
        mv.set_metric("quality_gate", "PASSED")
        mv.set_metric("champion_comparison", champ_note)
        return result

    _remove_candidate(session, version)
    raise RuntimeError(f"Quality gate FAILED for {MODEL_NAME}/{version}: " + "; ".join(failures))
