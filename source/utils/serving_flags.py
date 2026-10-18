"""Emit serving-mode flags from config for CI (GITHUB_OUTPUT).

Lets workflows skip the batch or real-time deployment steps when a mode is
disabled in source/configs/config.yml (serving.batch / serving.realtime).

Outputs: batch_enabled, realtime_enabled, monitor_enabled ("true"/"false")
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import serving_enabled


def flags() -> dict:
    batch = serving_enabled("batch")
    return {
        "batch_enabled": str(batch).lower(),
        "realtime_enabled": str(serving_enabled("realtime")).lower(),
        # The monitor reads batch predictions, so it requires batch scoring
        "monitor_enabled": str(batch).lower(),
    }


def main():
    values = flags()
    for k, v in values.items():
        print(f"{k}={v}")
    gh_output = os.getenv("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in values.items())


if __name__ == "__main__":
    main()
