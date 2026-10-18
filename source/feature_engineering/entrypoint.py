"""CLI entrypoint: register Feature Views in the Feature Store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering import feature_engineering
from snowpark_session import create_snowpark_session


def main():
    session = create_snowpark_session()
    result = feature_engineering.run(session)
    print(f"Feature engineering complete: {result}")


if __name__ == "__main__":
    main()
