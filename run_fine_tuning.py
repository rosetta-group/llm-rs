"""Run a JSON experiment config; imports never start training."""

import argparse
import json
from pathlib import Path

from configs.config_model import FineTuningConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON file matching FineTuningConfig")
    args = parser.parse_args()
    from fine_tuning import run_fine_tuning
    config = FineTuningConfig.model_validate_json(Path(args.config).read_text())
    print(json.dumps(run_fine_tuning(config), indent=2))


if __name__ == "__main__":
    main()
