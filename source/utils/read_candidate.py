"""Read the gated candidate produced by the training DAG (CI entry point).

The DAG's EVALUATE step has already applied the quality gate and registered
(or rejected) the candidate. This script publishes the decision to CI:
  - writes a Job Summary table (metrics, gate rules, champion comparison)
  - emits model_version=<V> to GITHUB_OUTPUT for downstream jobs
  - exits 1 if the gate did not pass

Reads METRICS_FILE, written by utils/wait_for_task.py.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, env_database

DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("stage"))
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")


def _rule_text(rule: dict) -> str:
    return " and ".join(f">= {rule['min']}" if k == "min" else f"<= {rule['max']}" for k in rule if k in ("min", "max"))


def summary_markdown(evaluation: dict) -> str:
    passed = evaluation.get("gate_passed")
    version = evaluation.get("model_version", "N/A")
    metrics = evaluation.get("metrics", {})
    rules = evaluation.get("gate_rules", {})
    champ = evaluation.get("champion", {}) or {}
    champ_metrics = champ.get("metrics", {}) if champ.get("available") else {}

    lines = [
        f"\n## Quality Gate: {'PASSED' if passed else 'FAILED'}\n",
        f"**Candidate**: `{DATABASE}.{SCHEMA}.{MODEL_NAME}/{version}`\n",
        f"**Champion**: {evaluation.get('champion_comparison', 'N/A')}\n",
        "| Metric | Candidate | Champion | Rule |",
        "|--------|-----------|----------|------|",
    ]
    for name, value in metrics.items():
        cand = f"{value:.4f}" if isinstance(value, float) else str(value)
        champ_v = champ_metrics.get(name)
        champ_s = f"{champ_v:.4f}" if isinstance(champ_v, float) else "—"
        lines.append(f"| {name} | {cand} | {champ_s} | {_rule_text(rules[name]) if name in rules else ''} |")
    for failure in evaluation.get("gate_failures", []):
        lines.append(f"\n- FAILED: {failure}")
    return "\n".join(lines) + "\n"


def main():
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"ERROR: evaluation result not found: {metrics_file}")
        sys.exit(1)
    with open(metrics_file) as f:
        evaluation = json.load(f)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(summary_markdown(evaluation))

    version = evaluation.get("model_version")
    if not evaluation.get("gate_passed") or not version:
        print("Quality gate did not pass — nothing to promote.")
        for failure in evaluation.get("gate_failures", []):
            print(f"  - {failure}")
        sys.exit(1)

    gh_output = os.getenv("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as fh:
            fh.write(f"model_version={version}\n")

    print("=" * 60)
    print(f"CANDIDATE READY: {MODEL_NAME}/{version}")
    print(f"  Registered in: {DATABASE}.{SCHEMA}")
    print(f"  Champion: {evaluation.get('champion_comparison')}")
    print("  PROD promotion happens in deploy-prod (after approval).")
    print("=" * 60)


if __name__ == "__main__":
    main()
