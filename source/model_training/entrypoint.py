"""CLI entrypoint: train the fraud detection model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training import training
from snowpark_session import create_snowpark_session


def main():
    session = create_snowpark_session()
    result = training.run(session)

    print("\n=== Training summary ===")
    metrics = result.get("metrics", {})
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
