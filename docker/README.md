# Docker

This directory contains Dockerfiles for running LeRobot in containerized environments. Both images are **built nightly from `main`** and published to Docker Hub with the full environment pre-baked — no dependency setup required.

## Pre-built Images

```bash
# CPU-only image (based on Dockerfile.user)
docker pull huggingface/lerobot-cpu:latest

# GPU image with CUDA support (based on Dockerfile.internal)
docker pull huggingface/lerobot-gpu:latest
```

## Quick Start

The fastest way to start training is to pull the GPU image and run `lerobot-train` directly. This is the same environment used for all of our CI, so it is a well-tested, batteries-included setup.

```bash
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest

# inside the container:
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Dockerfiles

### `Dockerfile.user` (CPU)

A lightweight image based on `python:3.12-slim`. Includes all Python dependencies and system libraries but does not include CUDA — there is no GPU support. Useful for exploring the codebase, running scripts, or working with robots, but not practical for training.

### `Dockerfile.internal` (GPU)

A CUDA-enabled image based on `nvidia/cuda`. This is the image for training — mostly used for internal interactions with the GPU cluster.

### `Dockerfile.train` (GPU, pip, rented-box training)

A CUDA-enabled image based on `nvidia/cuda`, installed with system pip (no uv) and an editable `lerobot` install. Default extras are `training,dataset_viz`; override with the `LEROBOT_EXTRAS` build arg (for example `training,dataset_viz,smolvla`). Not part of the nightly Docker Hub pipeline — built locally or on the box via `docker/docker-compose.train.yml`.

```bash
docker build -f docker/Dockerfile.train -t lerobot-train .
# or: make build-train
docker run -it --rm --gpus all --shm-size 16gb lerobot-train
```

### `Dockerfile.jetson` (NVIDIA Jetson, community-maintained)

Builds `torch`/`torchcodec`/`torchvision` from source with CUDA support for NVIDIA Jetson Orin (JetPack 6.2, CUDA 12.6) — no prebuilt cp312 wheel exists for this platform yet. **Not** part of the nightly CI/Docker Hub pipeline above: maintained by [@ravediamond](https://github.com/ravediamond), manually kept in sync with [`ravediamond/lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson) (the source of truth), where a prebuilt image is also published (`ghcr.io/ravediamond/lerobot-jetson`). See [#819](https://github.com/huggingface/lerobot/issues/819) for background.

```bash
docker build -f docker/Dockerfile.jetson -t lerobot-jetson .
docker run -it --rm --runtime nvidia lerobot-jetson
```

## Usage

### Running a pre-built image

```bash
# CPU
docker run -it --rm huggingface/lerobot-cpu:latest

# GPU
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest
```

### Building locally

From the repo root:

```bash
# CPU
docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user

# GPU
docker build -f docker/Dockerfile.internal -t lerobot-internal .
docker run -it --rm --gpus all --shm-size 16gb lerobot-internal
```

### Multi-GPU training

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` when launching the container:

```bash
# Use 4 GPUs
docker run -it --rm --gpus all --shm-size 16gb \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  huggingface/lerobot-gpu:latest
```

### USB device access (e.g. robots, cameras)

```bash
docker run -it --device=/dev/ -v /dev/:/dev/ --rm huggingface/lerobot-cpu:latest
```

## Training on a Lambda instance

`scripts/setup_lambda_train.sh` brings a rented Lambda GPU box to a runnable `lerobot-train` container. It rsyncs a local dataset, clones this repo on the box, builds `docker/Dockerfile.train` through `docker/docker-compose.train.yml`, checks CUDA and the dataset inside the container, and prints a `lerobot-train` command. It does not start training.

From the laptop, after the instance is up:

```bash
./scripts/setup_lambda_train.sh --ip <INSTANCE_IP> --ssh-key ~/.ssh/<lambda-key> \
  --dataset /path/to/<name>
```

The dataset basename becomes `/var/tmp/lerobot/datasets/<name>` on the box (override the root with `REMOTE_DATA_ROOT`) and `model_zoo/<name>/...` for training output. Re-run with `--skip-dataset --skip-build` once the box is already set up.

On the box, reach the container with:

```bash
ssh -A -i <key> ubuntu@<INSTANCE_IP>
tmux attach -t train
docker exec -it lerobot-train bash
```

Compose env overrides (also documented in the compose header): `DATA_ROOT`, `IMAGE_TAG`, `LEROBOT_EXTRAS`, `SHM_SIZE`, `HOST_UID` / `HOST_GID`, `CUDA_VISIBLE_DEVICES`, `WANDB_MODE`, `WANDB_API_KEY`. First run on the box: `mkdir -p model_zoo .train_home` so Docker does not create those dirs as root.
