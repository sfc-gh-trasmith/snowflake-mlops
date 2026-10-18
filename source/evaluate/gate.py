"""Quality gate rules: absolute metric bars and champion/challenger comparison.

Pure functions (no Snowflake calls) so they are unit-testable and reusable.
"""


def check_rules(metrics: dict, rules: dict) -> list[str]:
    """Check metrics against {name: {"min": x} | {"max": x}} rules.

    Returns a list of human-readable failures (empty = passed). A rule on a
    metric the model didn't produce is a failure, so a typo can't silently
    disable the gate.
    """
    failures = []
    for name, rule in rules.items():
        if name not in metrics or metrics[name] is None:
            failures.append(f"{name}: not produced by training (available: {sorted(metrics)})")
            continue
        value = float(metrics[name])
        if "min" in rule and value < float(rule["min"]):
            failures.append(f"{name} {value:.4f} < min {rule['min']}")
        if "max" in rule and value > float(rule["max"]):
            failures.append(f"{name} {value:.4f} > max {rule['max']}")
    return failures


def compare_champion(candidate: dict, champion: dict, cfg: dict) -> tuple[bool, str]:
    """Decide whether the candidate beats the champion.

    Args:
        candidate: candidate metrics
        champion: training-step champion result ({"available", "metrics", ...})
        cfg: quality_gate.champion config (metric, higher_is_better, min_improvement)

    Returns (passed, explanation). Passes when comparison is disabled or no
    champion is available (e.g. the first deployment).
    """
    if not cfg.get("enabled"):
        return True, "champion comparison disabled"
    if not champion or not champion.get("available"):
        return True, f"no champion to compare against ({(champion or {}).get('reason', 'unavailable')})"

    metric = cfg["metric"]
    if metric not in candidate or metric not in champion.get("metrics", {}):
        return False, f"champion metric '{metric}' missing from candidate or champion metrics"

    higher = bool(cfg.get("higher_is_better", True))
    margin = float(cfg.get("min_improvement", 0.0))
    cand, champ = float(candidate[metric]), float(champion["metrics"][metric])
    improvement = (cand - champ) if higher else (champ - cand)
    passed = improvement >= margin
    verdict = "beats" if passed else "does not beat"
    return passed, (
        f"candidate {metric}={cand:.4f} {verdict} champion {champion.get('version')} "
        f"{metric}={champ:.4f} (improvement {improvement:+.4f}, required >= {margin:+.4f})"
    )
