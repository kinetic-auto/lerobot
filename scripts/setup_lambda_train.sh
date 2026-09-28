#!/usr/bin/env bash
# Bring a Lambda on-demand box to a runnable lerobot-train container.
# Does not start lerobot-train. Does not fetch results.
#
# From the laptop, after the instance is up:
#   ./scripts/setup_lambda_train.sh --ip <INSTANCE_IP> --ssh-key ~/.ssh/<lambda-key> \
#     --dataset /path/to/<name>
#
# Options:
#   --ip IP              Lambda public IP (required unless --on-lambda)
#   --ssh-key PATH       SSH private key (optional if the agent already has it)
#   --user NAME          SSH user (default: ubuntu)
#   --dataset DIR        Local LeRobot dataset (required unless --skip-dataset)
#   --dataset-name NAME  Remote folder under datasets/ (required with --skip-dataset;
#                        defaults to the basename of --dataset)
#   --skip-dataset       Dataset is already at the remote path
#   --skip-build         compose up without --build
#   --on-lambda          Internal: run the on-box half (clone, docker, compose)
#   -h, --help
#
# Environment:
#   GIT_URL           default git@github.com:kinetic-auto/lerobot.git
#   GIT_REF           default main
#   REMOTE_REPO       default $HOME/github_repos/lerobot on the box
#   REMOTE_DATA_ROOT  default /var/tmp/lerobot
#   LEROBOT_EXTRAS    forwarded to the compose build (default: training,dataset_viz)

set -euo pipefail

GIT_URL="${GIT_URL:-git@github.com:kinetic-auto/lerobot.git}"
GIT_REF="${GIT_REF:-main}"
REMOTE_REPO="${REMOTE_REPO:-}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/var/tmp/lerobot}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,dataset_viz}"
SSH_USER="ubuntu"
INSTANCE_IP=""
SSH_KEY=""
LOCAL_DATASET_DIR=""
DATASET_NAME="${DATASET_NAME:-}"
DATASET_NAME_SET=false
SKIP_DATASET=false
SKIP_BUILD="${SKIP_BUILD:-false}"
ON_LAMBDA=false

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "[setup_lambda_train] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)
            INSTANCE_IP="${2:-}"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="${2:-}"
            shift 2
            ;;
        --user)
            SSH_USER="${2:-}"
            shift 2
            ;;
        --dataset)
            LOCAL_DATASET_DIR="${2:-}"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="${2:-}"
            DATASET_NAME_SET=true
            shift 2
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --on-lambda)
            ON_LAMBDA=true
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

if [[ -z "${DATASET_NAME}" && -n "${LOCAL_DATASET_DIR}" ]]; then
    DATASET_NAME="$(basename "${LOCAL_DATASET_DIR}")"
fi
REMOTE_DATASET_DIR="${REMOTE_DATA_ROOT}/datasets/${DATASET_NAME}"

ssh_cmd() {
    local args=(-A -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
    [[ -n "${SSH_KEY}" ]] && args+=(-i "${SSH_KEY}")
    ssh "${args[@]}" "${SSH_USER}@${INSTANCE_IP}" "$@"
}

rsync_cmd() {
    local args=(-aH --info=progress2)
    local ssh_args=(-A -o ServerAliveInterval=30)
    [[ -n "${SSH_KEY}" ]] && ssh_args+=(-i "${SSH_KEY}")
    rsync "${args[@]}" -e "ssh ${ssh_args[*]}" "$@"
}

sync_dataset_to_box() {
    log "rsync dataset -> ${REMOTE_DATASET_DIR}"
    rsync_cmd \
        "${LOCAL_DATASET_DIR}/" \
        "${SSH_USER}@${INSTANCE_IP}:${REMOTE_DATASET_DIR}/"
}

run_box_half() {
    local script_path="$1"
    ssh_cmd "GIT_URL=$(printf %q "${GIT_URL}") GIT_REF=$(printf %q "${GIT_REF}") DATASET_NAME=$(printf %q "${DATASET_NAME}") REMOTE_REPO=$(printf %q "${REMOTE_REPO}") REMOTE_DATA_ROOT=$(printf %q "${REMOTE_DATA_ROOT}") LEROBOT_EXTRAS=$(printf %q "${LEROBOT_EXTRAS}") SKIP_BUILD=$(printf %q "${SKIP_BUILD}") bash -s -- --on-lambda" \
        < "${script_path}"
}

orchestrate_from_laptop() {
    local script_path local_repo remote_ok remote_parent
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    local_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    [[ -n "${INSTANCE_IP}" ]] || die "--ip is required"
    [[ -f "${script_path}" ]] || die "cannot find ${script_path}"
    if [[ -n "${SSH_KEY}" && ! -f "${SSH_KEY}" ]]; then
        die "ssh key not found: ${SSH_KEY}"
    fi
    if [[ "${SKIP_DATASET}" == true ]]; then
        [[ "${DATASET_NAME_SET}" == true ]] || die "--skip-dataset requires --dataset-name"
    else
        [[ -n "${LOCAL_DATASET_DIR}" ]] || die "--dataset is required (or pass --skip-dataset)"
        [[ -f "${LOCAL_DATASET_DIR}/meta/info.json" ]] || die "local dataset missing: ${LOCAL_DATASET_DIR}/meta/info.json"
    fi

    REMOTE_REPO="${REMOTE_REPO:-/home/${SSH_USER}/github_repos/lerobot}"
    remote_parent="$(dirname "${REMOTE_REPO}")"

    log "ssh ${SSH_USER}@${INSTANCE_IP}"
    ssh_cmd 'echo connected; nvidia-smi -L'

    log "creating remote dirs"
    ssh_cmd "sudo mkdir -p $(printf %q "${REMOTE_DATA_ROOT}/datasets/${DATASET_NAME}") $(printf %q "${remote_parent}") && sudo chown -R $(printf %q "${SSH_USER}:${SSH_USER}") $(printf %q "${REMOTE_DATA_ROOT}") $(printf %q "${remote_parent}")"

    if [[ "${SKIP_DATASET}" == false ]]; then
        sync_dataset_to_box
    else
        log "skipping dataset rsync"
    fi

    log "on-box setup (clone, docker, compose)"
    remote_ok=0
    run_box_half "${script_path}" || remote_ok=$?

    if [[ "${remote_ok}" -ne 0 ]]; then
        log "git/docker setup failed (exit ${remote_ok}); rsyncing repo from laptop as fallback"
        rsync_cmd \
            --exclude 'model_zoo/' \
            --exclude '.train_home/' \
            --exclude 'outputs/' \
            --exclude 'wandb/' \
            --exclude '.venv/' \
            --exclude '.git/' \
            "${local_repo}/" \
            "${SSH_USER}@${INSTANCE_IP}:${REMOTE_REPO}/"
        run_box_half "${script_path}"
    fi
}

ensure_docker_compose() {
    if ! command -v docker >/dev/null; then
        die "docker is not installed (use the default Lambda Stack image)"
    fi
    if ! groups | grep -qw docker; then
        log "adding $(id -un) to docker group"
        sudo adduser "$(id -un)" docker
    fi
    if ! sg docker -c 'docker compose version' >/dev/null 2>&1; then
        log "installing docker compose plugin"
        sudo apt-get update -qq
        sudo apt-get install -y docker-compose-plugin
    fi
}

checkout_repo() {
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    if ! grep -q 'github.com' ~/.ssh/known_hosts 2>/dev/null; then
        ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    fi

    if [[ -d "${REMOTE_REPO}/.git" ]]; then
        log "repo present at ${REMOTE_REPO}"
        git -C "${REMOTE_REPO}" fetch origin && \
            git -C "${REMOTE_REPO}" checkout "${GIT_REF}" && \
            git -C "${REMOTE_REPO}" pull --ff-only origin "${GIT_REF}" || \
            log "git fetch/pull skipped (using files on disk)"
    elif [[ -f "${REMOTE_REPO}/docker/docker-compose.train.yml" ]]; then
        log "repo present at ${REMOTE_REPO} (no .git; using files on disk)"
    else
        log "cloning ${GIT_URL} (${GIT_REF}) -> ${REMOTE_REPO}"
        mkdir -p "$(dirname "${REMOTE_REPO}")"
        git clone --branch "${GIT_REF}" "${GIT_URL}" "${REMOTE_REPO}"
    fi
    log "HEAD $(git -C "${REMOTE_REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

verify_container() {
    local root="/data/datasets/${DATASET_NAME}"
    log "lerobot-info"
    sg docker -c "docker exec lerobot-train lerobot-info"
    log "dataset at ${root}"
    sg docker -c "docker exec lerobot-train python3 -c $(printf %q "from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata('local', root='${root}')
print(m.total_episodes, m.total_frames, m.fps, list(m.features))")"
}

print_train_hint() {
    cat <<EOF

Container is up. Training is NOT started.

  ssh -A -i <key> ${SSH_USER}@<INSTANCE_IP>
  tmux attach -t train
  docker exec -it lerobot-train bash

Then, inside the container.

Single GPU:

  lerobot-train \\
    --dataset.root=/data/datasets/${DATASET_NAME} \\
    --policy.type=act \\
    --job_name=act-chunk100-aug \\
    --dataset.image_transforms.enable=true \\
    --dataset.image_transforms.max_num_transforms=3 \\
    --policy.chunk_size=100 \\
    --policy.n_action_steps=100 \\
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \\
    --policy.kl_weight=10.0 \\
    --backbone=resnet18 \\
    --batch_size=8 \\
    --steps=100000 \\
    --save_freq=10000 \\
    --log_freq=200 \\
    --dataset.split_ratio='[0.9,0.0,0.1]' \\
    --accelerator.mixed_precision=bf16 \\
    --wandb.enable=false

Multi-GPU (DDP; set --num_processes to the GPU count):

  accelerate launch --multi_gpu --num_processes=2 \\
    \$(which lerobot-train) \\
    --dataset.root=/data/datasets/${DATASET_NAME} \\
    --policy.type=act \\
    --job_name=act-chunk100-aug \\
    --dataset.image_transforms.enable=true \\
    --dataset.image_transforms.max_num_transforms=3 \\
    --policy.chunk_size=100 \\
    --policy.n_action_steps=100 \\
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \\
    --policy.kl_weight=10.0 \\
    --backbone=resnet18 \\
    --batch_size=8 \\
    --steps=100000 \\
    --save_freq=10000 \\
    --log_freq=200 \\
    --dataset.split_ratio='[0.9,0.0,0.1]' \\
    --accelerator.mixed_precision=bf16 \\
    --wandb.enable=false

Runs land in model_zoo/${DATASET_NAME}/<job_name> on the box.
Use --batch_size=32 on a single A6000 48GB or A100 40GB.
EOF
}

prepare_lambda_box() {
    local uid gid build_flag
    uid="$(id -u)"
    gid="$(id -g)"
    GIT_REF="${GIT_REF:-main}"
    REMOTE_REPO="${REMOTE_REPO:-${HOME}/github_repos/lerobot}"
    REMOTE_DATASET_DIR="${REMOTE_DATA_ROOT}/datasets/${DATASET_NAME}"

    log "nvidia-smi"
    nvidia-smi

    log "host dirs"
    sudo mkdir -p "${REMOTE_DATA_ROOT}/datasets" "$(dirname "${REMOTE_REPO}")"
    sudo chown -R "$(id -un):$(id -gn)" "${REMOTE_DATA_ROOT}" "$(dirname "${REMOTE_REPO}")"

    ensure_docker_compose

    if [[ ! -f "${REMOTE_DATASET_DIR}/meta/info.json" ]]; then
        die "dataset missing at ${REMOTE_DATASET_DIR} (rsync it from the laptop first)"
    fi

    checkout_repo

    cd "${REMOTE_REPO}"
    mkdir -p model_zoo .train_home
    if ! [[ -O model_zoo && -O .train_home ]]; then
        log "fixing ownership on writable repo dirs (uid=${uid} gid=${gid})"
        sudo chown -R "${uid}:${gid}" model_zoo .train_home
    fi

    tmux has-session -t train 2>/dev/null || tmux new-session -d -s train

    build_flag=""
    [[ "${SKIP_BUILD}" != true ]] && build_flag="--build"
    log "compose up lerobot-train ${build_flag}"
    sg docker -c "HOST_UID=${uid} HOST_GID=${gid} LEROBOT_EXTRAS=$(printf %q "${LEROBOT_EXTRAS}") docker compose -f docker/docker-compose.train.yml up -d ${build_flag}"

    verify_container
    print_train_hint
}

if [[ "${ON_LAMBDA}" == true ]]; then
    [[ -n "${DATASET_NAME}" ]] || die "DATASET_NAME is required on the box"
    prepare_lambda_box
else
    orchestrate_from_laptop
fi
