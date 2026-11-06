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
"""Scan a session of MCAP episodes for recording problems.

Requires: pip install 'lerobot[mcap-dep]'

Example:
```
lerobot-mcap-valid \
    --session_directory /path/to/session \
    --d405_depth_intrinsics_path /path/to/d405_depth_intrinsics.yaml \
    --fail_on_critical true
```
"""

import logging
from pathlib import Path

from lerobot.configs import parser
from lerobot.datasets.mcap_validation import (
    REPORT_DIRECTORY_NAME,
    EpisodeReport,
    McapValidationConfig,
    Severity,
    validate_session,
    write_reports,
)
from lerobot.utils.utils import init_logging


@parser.wrap()
def run_mcap_validation(config: McapValidationConfig) -> None:
    episode_reports = validate_session(config)
    report_directory = (
        Path(config.report_directory)
        if config.report_directory is not None
        else Path(config.session_directory) / REPORT_DIRECTORY_NAME
    )
    write_reports(episode_reports, report_directory)
    print_summary(episode_reports)
    logging.info("Wrote the validation report to %s", report_directory)

    has_critical_episode = any(report.severity == Severity.CRITICAL for report in episode_reports)
    if config.fail_on_critical and has_critical_episode:
        raise SystemExit(1)


def print_summary(episode_reports: list[EpisodeReport]) -> None:
    for report in episode_reports:
        issue_messages = "; ".join(issue.message for issue in report.all_issues)
        print(f"{report.severity.value:<8} {report.episode_id}  {issue_messages}")
    for severity in Severity:
        count = sum(report.severity == severity for report in episode_reports)
        print(f"{severity.value}: {count}")


def main() -> None:
    init_logging()
    run_mcap_validation()


if __name__ == "__main__":
    main()
