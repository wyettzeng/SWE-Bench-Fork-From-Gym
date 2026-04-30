#! /usr/bin/env bash

env \
    XDG_CACHE_HOME="$HOME/.cache" \
    HOME="$HOME" \
    TMPDIR="/tmp" \
    podman system service -t 0 &
export DOCKER_HOST=unix://tmp/podman-run-68344/podman/podman.sock
export CONTAINER_HOST=$DOCKER_HOST
unshare -r


python -m swebench.harness.eval_server \
    --dataset_name SWE-Gym/SWE-Gym \
    --split train \
    --max_workers 16 \
    --remote_image_namespace docker.io/xingyaoww \
    --log_dir "$SCRATCH_DISK/evaluator_logs2/" \
    --port 8080
