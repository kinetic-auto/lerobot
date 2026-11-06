#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate a converted LeRobotDataset.

Example:
```
lerobot-dataset-valid \
    --dataset-path /path/to/my_dataset
```
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_validation import validate_dataset
from lerobot.utils.utils import init_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Local dataset folder path.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/dataset",
        help="Repository id the dataset was created with (default: local/dataset).",
    )
    arguments = parser.parse_args()

    init_logging()

    results = validate_dataset(Path(arguments.dataset_path), arguments.repo_id)

    for result in results:
        status = "OK" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
