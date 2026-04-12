#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time

from argparse import ArgumentParser
from pathlib import Path

import requests


def load_predictions(predictions_path: Path) -> list[dict]:
    if predictions_path.suffix == ".jsonl":
        return [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]

    data = json.loads(predictions_path.read_text())
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported predictions format in {predictions_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--predictions_path", type=Path, required=True, help="Path to preds.json or preds.jsonl")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8080", help="Evaluation server base URL")
    parser.add_argument("--num_instances", type=int, default=20, help="Number of random instances to test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--poll_interval", type=float, default=5.0, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-job evaluation timeout sent to the server")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions_path)
    if len(predictions) < args.num_instances:
        raise ValueError(
            f"Requested {args.num_instances} instances but only found {len(predictions)} predictions"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(predictions, args.num_instances)

    print(f"Loaded {len(predictions)} predictions from {args.predictions_path}")
    print(f"Submitting {len(sampled)} random instances to {args.server_url}")

    jobs: dict[str, dict] = {}
    for pred in sampled:
        payload = {
            "instance_id": pred["instance_id"],
            "model_patch": pred.get("model_patch", ""),
            "model_name_or_path": pred.get("model_name_or_path", "eval_server_test"),
            "timeout": args.timeout,
        }
        response = requests.post(f"{args.server_url.rstrip('/')}/evaluate", json=payload, timeout=30)
        response.raise_for_status()
        job = response.json()
        jobs[job["job_id"]] = {
            "instance_id": pred["instance_id"],
            "status": job["status"],
        }
        print(f"submitted {pred['instance_id']} -> job_id={job['job_id']} status={job['status']}")

    pending = set(jobs)
    while pending:
        print(f"polling {len(pending)} pending jobs...")
        finished_this_round = []
        for job_id in list(pending):
            response = requests.get(f"{args.server_url.rstrip('/')}/jobs/{job_id}", timeout=30)
            response.raise_for_status()
            job = response.json()
            jobs[job_id] = job
            status = job["status"]
            if status in {"completed", "failed"}:
                finished_this_round.append(job_id)
                instance_id = job["instance_id"]
                if status == "completed":
                    result = job["result"]
                    print(
                        f"[completed] {instance_id} "
                        f"resolved={result['resolved']} "
                        f"passed={result['passed_count']} "
                        f"failed={result['failed_count']}"
                    )
                else:
                    print(f"[failed] {instance_id} error={job.get('error')}")

        for job_id in finished_this_round:
            pending.remove(job_id)

        if pending:
            time.sleep(args.poll_interval)

    print("\nfinal results:")
    completed = 0
    failed = 0
    resolved = 0
    for job in jobs.values():
        if job["status"] == "completed":
            completed += 1
            resolved += int(bool(job["result"]["resolved"]))
        else:
            failed += 1

    print(f"completed={completed} failed={failed} resolved={resolved}")
    for job in jobs.values():
        instance_id = job["instance_id"]
        status = job["status"]
        if status == "completed":
            result = job["result"]
            print(
                f"{instance_id}: status={status} resolved={result['resolved']} "
                f"passed_tests={result['passed_tests']} failed_tests={result['failed_tests']}"
            )
        else:
            print(f"{instance_id}: status={status} error={job.get('error')}")


if __name__ == "__main__":
    main()
