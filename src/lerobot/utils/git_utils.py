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

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def git_provenance(repository_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    if not (repository_root / ".git").exists():
        return {}

    result: dict[str, str] = {}
    commands = {
        "code_commit": ["git", "rev-parse", "HEAD"],
        "code_tag": ["git", "describe", "--tags", "--exact-match", "HEAD"],
    }
    for key, command in commands.items():
        try:
            value = subprocess.check_output(
                command,
                cwd=repository_root,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if value:
            result[key] = value
    return result
