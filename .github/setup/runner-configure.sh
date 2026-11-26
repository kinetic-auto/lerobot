#!/usr/bin/env bash
#
# Register the kinetic-auto/lerobot runner service on supertower.
# Bootstraps .github/setup/venv, then runs _playbook_configure.
#
# Does not install Docker or change host configuration. SSH to
# kinetic@192.168.1.219 must already work with ~/.ssh/id_ed25519.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found." >&2
  echo "Install Python 3 on this control machine." >&2
  exit 1
fi

if ! python3 -c "import venv" 2>/dev/null; then
  echo "Error: python3 venv module not available." >&2
  echo "Install: sudo apt install python3-venv" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh not found." >&2
  echo "Install the GitHub CLI and authenticate with access to kinetic-auto/lerobot." >&2
  exit 1
fi

if [ ! -f "${REQ_FILE}" ]; then
  echo "Error: requirements file not found at ${REQ_FILE}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "===> Creating control-machine venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "===> Ensuring Python dependencies in ${VENV_DIR}"
"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q -r "${REQ_FILE}"

export PATH="${VENV_DIR}/bin:${PATH}"
cd "${SCRIPT_DIR}"
exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/_runner_configure.py"
