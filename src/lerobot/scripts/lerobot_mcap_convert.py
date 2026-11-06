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
"""Convert a validated session of MCAP episodes into a LeRobotDataset v3.0.

Requires: pip install 'lerobot[mcap-dep]' and a report written by `lerobot-mcap-valid` for the session.

Example:
```
lerobot-mcap-convert \
    --session_directory /path/to/session \
    --output_root /path/to/new_dataset \
    --repo_id local/door_unlatch \
    --d405_color_intrinsics_path /path/to/d405_intrinsics.yaml \
    --d405_depth_intrinsics_path /path/to/d405_depth_intrinsics.yaml

lerobot-mcap-convert \
    --session_directory /path/to/sim_session \
    --output_root /path/to/new_dataset \
    --repo_id local/door_unlatch_sim \
    --align_depth false
```
"""

from lerobot.configs import parser
from lerobot.datasets.mcap_conversion import McapConversionConfig, convert_session
from lerobot.utils.utils import init_logging


@parser.wrap()
def run_mcap_conversion(config: McapConversionConfig) -> None:
    convert_session(config)


def main() -> None:
    init_logging()
    run_mcap_conversion()


if __name__ == "__main__":
    main()
