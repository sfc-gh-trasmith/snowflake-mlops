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
from config import CHAMPION_CONFIG, MODEL_NAME, env_database

DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("stage"))
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")

# Metrics where a lower value is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"log_loss", "rmse", "mae", "mse", "mape"}


def _rule_text(rule: dict) -> str:
    return " and ".join(f">= {rule['min']}" if k == "min" else f"<= {rule['max']}" for k in rule if k in ("min", "max"))


def _rule_ok(value, rule: dict) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return ("min" not in rule or value >= rule["min"]) and ("max" not in rule or value <= rule["max"])


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else ("—" if value is None else str(value))


def _delta(name: str, cand, champ) -> str:
    """Signed candidate-minus-champion change, labelled better/worse for the metric."""
    if not isinstance(cand, (int, float)) or not isinstance(champ, (int, float)):
        return "—"
    diff = cand - champ
    if abs(diff) < 5e-5:
        return f"{diff:+.4f} (same)"
    better = diff < 0 if name in LOWER_IS_BETTER else diff > 0
    return f"{diff:+.4f} ({'better' if better else 'worse'})"


def summary_markdown(evaluation: dict) -> str:
    """GitHub Job Summary: candidate vs PROD champion, gate rules, lineage, next step."""
    passed = evaluation.get("gate_passed")
    version = evaluation.get("model_version", "N/A")
    metrics = evaluation.get("metrics", {})
    rules = evaluation.get("gate_rules", {})
    champ = evaluation.get("champion", {}) or {}
    has_champ = bool(champ.get("available"))
    champ_metrics = champ.get("metrics", {}) if has_champ else {}
    champ_label = f"Champion ({champ.get('version')})" if has_champ else "Champion"
    decision_metric = CHAMPION_CONFIG.get("metric") if CHAMPION_CONFIG.get("enabled") else None

    lines = [
        f"\n## STAGE candidate `{MODEL_NAME}/{version}`: quality gate {'PASSED' if passed else 'FAILED'}\n",
        f"**Champion comparison:** {evaluation.get('champion_comparison', 'N/A')}\n",
    ]
    if has_champ:
        lines.append(
            f"Both models were scored on the **same holdout**; the champion is the default version of "
            f"`{champ.get('source')}`. Decision metric: `{decision_metric}`.\n"
        )
    else:
        lines.append("No champion in PROD yet (first deployment), so only the metric rules apply.\n")

    lines += [
        f"| Metric | Candidate ({version}) | {champ_label} | Change | Gate rule | Result |",
        "|--------|---------:|---------:|--------|-----------|--------|",
    ]
    for name, value in metrics.items():
        label = f"**{name}** (decision)" if name == decision_metric else name
        rule = rules.get(name)
        result = ("pass" if _rule_ok(value, rule) else "**FAIL**") if rule else ""
        lines.append(
            f"| {label} | {_fmt(value)} | {_fmt(champ_metrics.get(name))} | "
            f"{_delta(name, value, champ_metrics.get(name))} | {_rule_text(rule) if rule else ''} | {result} |"
        )

    if evaluation.get("gate_failures"):
        lines.append("\n**Gate failures:**")
        lines += [f"- {failure}" for failure in evaluation["gate_failures"]]

    rows = evaluation.get("dataset_rows")
    lines += [
        "\n<details><summary>Lineage</summary>\n",
        "| | |",
        "|---|---|",
        f"| Candidate | `{DATABASE}.{SCHEMA}.{MODEL_NAME}/{version}` |",
        f"| Feature view | `{evaluation.get('feature_view') or 'N/A'}` |",
        f"| Training dataset | `{evaluation.get('dataset') or 'N/A'}`" + (f" ({rows:,} rows)" if rows else "") + " |",
        f"| Commit | `{evaluation.get('git_sha') or 'N/A'}` |",
        "\n</details>\n",
    ]

    if passed:
        lines.append(f"**Next:** approve the `PROD` environment to promote `{version}`.")
    else:
        lines.append("**Next:** the candidate was removed from the STAGE registry; nothing will be promoted.")
    return "\n".join(lines) + "\n"


def failure_markdown(evaluation: dict) -> str:
    """Summary when the DAG failed before EVALUATE produced a result."""
    lines = [f"\n## STAGE training pipeline FAILED: no candidate for `{MODEL_NAME}`\n"]
    for task, error in (evaluation.get("failed_tasks") or {}).items():
        lines.append(f"- `{task}`: {error}")
    if not evaluation.get("failed_tasks"):
        lines.append("- No evaluation result was recorded for this run; see the wait step's log.")
    return "\n".join(lines) + "\n"


def main():
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    evaluation = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            evaluation = json.load(f)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(summary_markdown(evaluation) if evaluation.get("metrics") else failure_markdown(evaluation))

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
