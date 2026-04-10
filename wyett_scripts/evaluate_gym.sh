#! /usr/bin/env bash

set -euo pipefail

# podman system service -t 0 &
# export DOCKER_HOST=unix://tmp/podman-run-68344/podman/podman.sock
# export CONTAINER_HOST=$DOCKER_HOST
# unshare -r

export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
export CONTAINER_HOST="$DOCKER_HOST"

podman system service --time=0 &

export RUN_DIR="$SCRATCH_DISK/runs/swegym_pandas_qwen"


python -m swebench.harness.run_evaluation \
    --dataset_name SWE-Gym/SWE-Gym \
    --predictions_path "${RUN_DIR}/preds.json" \
    --split train \
    --max_workers 12 \
    --run_id gym_pandas_qwen3
