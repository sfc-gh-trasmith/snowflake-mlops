"""CLI entrypoint: evaluate the latest training run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluation
from snowpark_session import create_snowpark_session


def main():
    session = create_snowpark_session()
    result = evaluation.run(session)

    print("\n=== Evaluation summary ===")
    metrics = result.get("metrics", {})
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"  pipeline_status: {result.get('pipeline_status')}")


if __name__ == "__main__":
    main()
