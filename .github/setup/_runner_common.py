"""Shared helpers for the lerobot runner setup (control machine only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

CONTROL_SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
CONTROL_SSH_KEY_PUB = CONTROL_SSH_KEY.with_name(f"{CONTROL_SSH_KEY.name}.pub")


def require_control_ssh_key() -> None:
    if CONTROL_SSH_KEY.exists() and CONTROL_SSH_KEY_PUB.exists():
        return
    raise RuntimeError(
        f"Control-machine SSH key not found at {CONTROL_SSH_KEY} "
        f"(and {CONTROL_SSH_KEY_PUB})."
    )


def control_can_ssh_to_target(ansible_host: str, user: str = "kinetic") -> bool:
    """Return whether this machine can SSH to the target with the default key."""
    cmd = [
        "ssh",
        "-i",
        str(CONTROL_SSH_KEY),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        f"{user}@{ansible_host}",
        "echo",
        "OK",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.returncode == 0
