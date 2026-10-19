"""Tests for promotion, the CI job summary, retention, and automatic rollback."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "deploy"))


# ─── Promotion idempotency ──────────────────────────────────────────────────


def _version(comment="", size=100, metadata="{}"):
    return {"comment": comment, "size": size, "metadata": metadata}


def test_promotion_recognizes_same_training_run():
    from promote import same_version

    stage = _version("dataset:D/V1_20261018_120000 | git:abc1234")
    assert same_version(_version("dataset:D/V1_20261018_120000 | git:abc1234"), stage)
    # A rebuilt STAGE reuses the name V1 for a different run: must not match.
    assert not same_version(_version("dataset:D/V1_20261019_090000 | git:def5678"), stage)


def test_promotion_falls_back_for_copies_without_comment():
    from promote import same_version

    stage = _version("dataset:D/V1_x", size=268318, metadata='{"m": 1}')
    assert same_version(_version("", size=268318, metadata='{"m": 1}'), stage)
    assert not same_version(_version("", size=1, metadata='{"m": 1}'), stage)


def test_comment_literal_escapes_quotes():
    from promote import _sql_string

    assert _sql_string("it's") == "'it\\'s'"
    assert _sql_string(None) == "''"


# ─── CI job summary ─────────────────────────────────────────────────────────


def _evaluation(**overrides):
    ev = {
        "model_version": "V5",
        "gate_passed": True,
        "metrics": {"auc_roc": 0.91, "log_loss": 0.30, "recall": 0.70},
        "gate_rules": {"auc_roc": {"min": 0.8}, "recall": {"min": 0.75}},
        "champion": {
            "available": True,
            "version": "V4",
            "source": "PROD.ML.M",
            "metrics": {"auc_roc": 0.88, "log_loss": 0.35, "recall": 0.72},
        },
        "champion_comparison": "candidate auc_roc=0.9100 beats champion V4 auc_roc=0.8800",
        "dataset": "D/V1_20261019_000000",
        "feature_view": "FV$V1",
        "git_sha": "abc1234",
    }
    ev.update(overrides)
    return ev


def test_summary_compares_candidate_with_champion():
    from utils.read_candidate import summary_markdown

    md = summary_markdown(_evaluation())
    assert "Champion (V4)" in md and "quality gate PASSED" in md
    assert "+0.0300 (better)" in md  # auc_roc up
    assert "-0.0500 (better)" in md  # log_loss down is better
    assert "| **FAIL** |" in md  # recall 0.70 < 0.75 rule
    assert "`D/V1_20261019_000000`" in md and "approve the `PROD` environment" in md


def test_summary_without_champion_and_on_rejection():
    from utils.read_candidate import summary_markdown

    md = summary_markdown(
        _evaluation(champion={"available": False}, gate_passed=False, gate_failures=["recall=0.7000 < min 0.75"])
    )
    assert "No champion in PROD yet" in md and "quality gate FAILED" in md
    assert "recall=0.7000 < min 0.75" in md and "removed from the STAGE registry" in md


def test_failure_summary_names_failed_task():
    from utils.read_candidate import failure_markdown

    md = failure_markdown({"failed_tasks": {"MLOPS_DEMO_TASK$TRAIN_MODEL": "Job failed"}})
    assert "TRAIN_MODEL" in md and "FAILED" in md


# ─── Retention ────────────────────────────────────────────────────────────────


def test_retention_keeps_newest_and_protected_versions():
    from utils.retention import plan_model_versions

    versions = [{"name": f"V{i}"} for i in (1, 2, 10, 3, 4, 5)]
    # Numeric order (V10 is newest), keep 2, V1 is the default.
    assert plan_model_versions(versions, 2, protected={"V1"}) == ["V4", "V3", "V2"]
    assert plan_model_versions(versions, 10, protected=set()) == []


def test_retention_keeps_datasets_referenced_by_lineage():
    from utils.retention import dataset_refs, plan_dataset_versions

    refs = dataset_refs(["features:FV$V1 | dataset:D/V1_20261001_000000 | git:abc", "", None])
    assert refs == {("D", "V1_20261001_000000")}
    versions = ["V1_20261001_000000", "V2_20261003_000000", "V1_20261002_000000", "V2_20261004_000000"]
    referenced = {v for d, v in refs if d == "D"}
    # Sorted by timestamp, not by feature view version prefix.
    assert plan_dataset_versions(versions, 2, referenced) == ["V1_20261002_000000"]


def test_retention_runs_after_successful_deploys():
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert "retention.py --env stage" in workflow and "retention.py --env prod" in workflow


# ─── Automatic rollback ───────────────────────────────────────────────────────


def test_gateway_split_round_trips_to_targets_arg():
    from apply_gateway_config import parse_targets_arg
    from prod_state import targets_arg

    targets = parse_targets_arg("V4:80,V5:20")
    assert targets_arg(targets) == "V4:80,V5:20"


def test_restore_without_previous_version_is_a_noop():
    from prod_state import restore

    class NoSQL:
        def sql(self, _):
            raise AssertionError("first deployment must not touch PROD")

    restore(NoSQL(), "", "")


def test_deploy_prod_smoke_tests_then_rolls_back_on_failure():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text())
    steps = workflow["jobs"]["deploy-prod"]["steps"]
    names = [s.get("name", "") for s in steps]
    idx = {
        key: next(i for i, n in enumerate(names) if key in n)
        for key in ("Capture", "Promote", "gateway traffic", "smoke", "Open PR syncing", "rollback")
    }
    assert (
        idx["Capture"]
        < idx["Promote"]
        < idx["gateway traffic"]
        < idx["smoke"]
        < idx["Open PR syncing"]
        < idx["rollback"]
    )
    rollback = steps[idx["rollback"]]
    assert rollback["if"] == "failure() && steps.promote.outcome == 'success'"
    assert "prod_state.py --restore" in rollback["run"]
