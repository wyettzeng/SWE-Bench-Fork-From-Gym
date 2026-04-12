#! /usr/bin/env bash

set -euo pipefail

env \
    XDG_CACHE_HOME="$HOME/.cache" \
    HOME="$HOME" \
    TMPDIR="/tmp" \
    podman system service -t 0 &
export DOCKER_HOST=unix://tmp/podman-run-68344/podman/podman.sock
export CONTAINER_HOST=$DOCKER_HOST
unshare -r

rm -rf logs

python -m swebench.harness.eval_server \
    --dataset_name SWE-Gym/SWE-Gym \
    --split train \
    --max_workers 12 \
    --remote_image_namespace docker.io/xingyaoww
