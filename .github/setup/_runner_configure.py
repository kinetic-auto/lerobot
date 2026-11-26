#!/usr/bin/env python3
"""Confirm, then register the lerobot runner service on supertower.

Invoked by runner-configure.sh, which bootstraps .github/setup/venv first.
"""

import os
import subprocess
import sys
from pathlib import Path

import questionary
import yaml

from _runner_common import CONTROL_SSH_KEY, control_can_ssh_to_target, require_control_ssh_key

SCRIPT_DIR = Path(__file__).parent
INVENTORY_PATH = SCRIPT_DIR / "inventory" / "runners.yaml"
PLAYBOOK_PATH = SCRIPT_DIR / "_playbook_configure.yaml"
RUNNER_NAME = "supertower"


def load_target() -> str:
    with open(INVENTORY_PATH) as handle:
        hosts = yaml.safe_load(handle)["all"]["hosts"]
    if list(hosts) != [RUNNER_NAME]:
        raise RuntimeError(f"inventory must contain only {RUNNER_NAME}")
    ansible_host = hosts[RUNNER_NAME].get("ansible_host")
    if not ansible_host:
        raise RuntimeError(f"{RUNNER_NAME} is missing ansible_host")
    return str(ansible_host)


def run_playbook():
    if not PLAYBOOK_PATH.exists():
        raise RuntimeError(f"Playbook not found at {PLAYBOOK_PATH}")

    env = os.environ.copy()
    env["ANSIBLE_HOST_KEY_CHECKING"] = "False"
    cmd = [
        "ansible-playbook",
        str(PLAYBOOK_PATH),
        "-v",
        "-i",
        str(INVENTORY_PATH),
        "--limit",
        RUNNER_NAME,
        "-e",
        f"ansible_ssh_private_key_file={CONTROL_SSH_KEY}",
    ]
    print("\nRunning _playbook_configure\n")
    return subprocess.run(cmd, env=env, cwd=SCRIPT_DIR)


def main() -> None:
    try:
        print("=== lerobot runner on supertower ===\n")
        require_control_ssh_key()
        ansible_host = load_target()

        proceed = questionary.confirm(
            "Register the kinetic-auto/lerobot runner on supertower?",
            default=True,
        ).unsafe_ask()
        if not proceed:
            print("Cancelled.")
            return

        print(f"Verifying SSH to kinetic@{ansible_host}...")
        if not control_can_ssh_to_target(ansible_host):
            raise RuntimeError(
                f"SSH preflight failed for kinetic@{ansible_host} "
                f"({CONTROL_SSH_KEY})."
            )
        print("SSH reachability OK.\n")
        sys.exit(run_playbook().returncode)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
