#!/usr/bin/env bash
# Build the local train image and run the stub Slack loop on this machine.
# No GPU and no dataset. The container posts one parent message and edits one thread reply.
#
# Fill in SLACK_BOT_TOKEN and SLACK_CHANNEL below, then:
#   ./scripts/stub_train_slack.sh
#
# Options:
#   --skip-build    Reuse lerobot-train:${IMAGE_TAG} instead of building
#   -h, --help
#
# Environment:
#   IMAGE_TAG         default latest
#   LEROBOT_EXTRAS    pip extras at build (default: training,dataset_viz,transformers-dep)
#   HOST_UID          default: current uid
#   HOST_GID          default: current gid
#   SKIP_BUILD        set to true to skip the build (same as --skip-build)

set -euo pipefail

# Fill these in. Passed to the container only; not baked into the image.
SLACK_BOT_TOKEN=""
SLACK_CHANNEL=""

IMAGE_TAG="${IMAGE_TAG:-latest}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,dataset_viz,transformers-dep}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
SKIP_BUILD="${SKIP_BUILD:-false}"
IMAGE="lerobot-train:${IMAGE_TAG}"

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "[stub_train_slack] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "${SLACK_BOT_TOKEN}" != "xoxb-replace-me" ]] || die "set SLACK_BOT_TOKEN in this script"
[[ "${SLACK_CHANNEL}" != "C0123456789" ]] || die "set SLACK_CHANNEL in this script"
command -v docker >/dev/null || die "docker is not installed"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SKIP_BUILD}" != true ]]; then
    log "docker build -f docker/Dockerfile.train -t ${IMAGE}"
    docker build \
        -f docker/Dockerfile.train \
        --build-arg "LEROBOT_EXTRAS=${LEROBOT_EXTRAS}" \
        --build-arg "HOST_UID=${HOST_UID}" \
        --build-arg "HOST_GID=${HOST_GID}" \
        -t "${IMAGE}" \
        .
else
    log "skipping docker build"
fi

log "running stub in ${IMAGE}"
docker run --rm \
    -e "SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN}" \
    -e "SLACK_CHANNEL=${SLACK_CHANNEL}" \
    -e HOME=/tmp \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -u "${HOST_UID}:${HOST_GID}" \
    -v "${REPO_ROOT}/src:/lerobot/src:ro" \
    -w /lerobot \
    "${IMAGE}" \
    python -m lerobot.scripts.stub_train_slack
