"""Notify: create GitHub issue when a retrained model candidate is ready for review.

Called after scheduled retraining completes (STAGE_ONLY mode).
Creates a GitHub issue with metrics so a human can review and approve promotion.

Requires: GH_TOKEN environment variable (GitHub Actions provides this automatically).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, env_database

DATABASE = os.getenv("SNOWFLAKE_DATABASE", env_database("stage"))
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
REPO = os.getenv("GITHUB_REPOSITORY", "<your-org>/snowflake-mlops")


def _metrics_table(metrics: dict) -> str:
    rows = ["| Metric | Value |", "|--------|-------|"]
    for name, value in metrics.items():
        rows.append(f"| {name} | {value:.4f} |" if isinstance(value, float) else f"| {name} | {value} |")
    return "\n".join(rows)


def create_github_issue(version: str, evaluation: dict):
    """Create a GitHub issue notifying that a candidate model is ready."""
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        print("  WARNING: No GH_TOKEN set — skipping GitHub issue creation.")
        return None

    metrics = evaluation.get("metrics", {})
    title = f"Model Candidate Ready: {MODEL_NAME}/{version}"
    body = f"""## Scheduled Retrain Complete

A new model candidate has been trained and registered in STAGE.

**Model:** `{DATABASE}.{SCHEMA}.{MODEL_NAME}`
**Version:** `{version}`
**Status:** Awaiting human approval for PROD promotion

### Metrics

{_metrics_table(metrics)}

**Champion:** {evaluation.get("champion_comparison", "N/A")}

### To Promote to PROD

1. Review the metrics above
2. Go to **Actions** > **Deploy** > **Run workflow** (manual dispatch)
3. The deploy-prod job will promote this version to production

### To Reject

Close this issue — the candidate stays in STAGE and won't be promoted.
"""

    url = f"https://api.github.com/repos/{REPO}/issues"
    data = json.dumps({"title": title, "body": body, "labels": ["model-candidate", "retrain"]}).encode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"  GitHub issue created: {result['html_url']}")
            return result["html_url"]
    except Exception as e:
        print(f"  WARNING: Failed to create GitHub issue: {e}")
        return None


def main():
    metrics_file = os.getenv("METRICS_FILE", "/tmp/pipeline_metrics.json")
    if not os.path.exists(metrics_file):
        print("ERROR: No metrics file found. Cannot notify.")
        sys.exit(1)

    with open(metrics_file) as f:
        evaluation = json.load(f)

    version = evaluation.get("model_version", "unknown")

    print("=" * 60)
    print("NOTIFY: Model Candidate Ready")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}/{version}")
    print(f"  Champion: {evaluation.get('champion_comparison', 'N/A')}")

    create_github_issue(version, evaluation)

    print("=" * 60)


if __name__ == "__main__":
    main()
