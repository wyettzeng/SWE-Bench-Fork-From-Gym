#! /usr/bin/env bash

set -euo pipefail

# env \
#     XDG_CACHE_HOME="$HOME/.cache" \
#     HOME="$HOME" \
#     TMPDIR="/tmp" \
#     podman system service -t 0 &
podman system service -t 0 &
export DOCKER_HOST=unix://tmp/podman-run-68344/podman/podman.sock
export CONTAINER_HOST=$DOCKER_HOST
unshare -r

rm -rf logs

export RUN_DIR="$SCRATCH_DISK/runs/swegym_pandas_qwen"
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-Gym/SWE-Gym \
    --predictions_path "${RUN_DIR}/preds.json" \
    --split train \
    --max_workers 12 \
    --run_id gym_pandas_qwen3 \
    --use_remote_instance_images true \
    --remote_instance_image_namespace docker.io/xingyaoww
