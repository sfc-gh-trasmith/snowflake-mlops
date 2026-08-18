"""Model promotion dispatcher.

Routes model promotion to the correct strategy based on the TOPOLOGY
environment variable (or --strategy CLI arg).

Supported topologies:
  - single-account (default): Cross-database replication within one account
  - multi-account: Cross-account model sharing (same region)
  - cross-region: Cross-account + cross-region replication

Usage:
  python deploy/promote.py --version V3
  python deploy/promote.py --version V3 --strategy multi-account
  TOPOLOGY=single-account python deploy/promote.py --version V3
"""

import argparse
import os
import sys


STRATEGIES = {
    "single-account": "deploy.strategies.single_account",
    "multi-account": "deploy.strategies.multi_account",
    "cross-region": "deploy.strategies.cross_region",
}

DEFAULT_TOPOLOGY = "single-account"


def get_strategy(topology: str):
    """Dynamically import and return the promote function for a topology."""
    if topology not in STRATEGIES:
        print(f"ERROR: Unknown topology '{topology}'")
        print(f"  Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    module_path = STRATEGIES[topology]
    # Import the module
    import importlib

    module = importlib.import_module(module_path)
    return module.promote


def main():
    parser = argparse.ArgumentParser(description="Promote a model version to PROD")
    parser.add_argument("--version", required=True, help="Model version to promote (e.g., V3)")
    parser.add_argument(
        "--strategy",
        default=os.getenv("TOPOLOGY", DEFAULT_TOPOLOGY),
        choices=STRATEGIES.keys(),
        help=f"Promotion topology (default: $TOPOLOGY or '{DEFAULT_TOPOLOGY}')",
    )
    args = parser.parse_args()

    print("Model Promotion")
    print(f"  Version:  {args.version}")
    print(f"  Strategy: {args.strategy}")
    print()

    promote_fn = get_strategy(args.strategy)
    promote_fn(version=args.version)

    print("\nPromotion complete.")


if __name__ == "__main__":
    main()
