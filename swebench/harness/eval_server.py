from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid

from argparse import ArgumentParser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Full, Queue
from typing import Any
from urllib.parse import urlparse

import docker

from swebench.harness.constants import KEY_INSTANCE_ID, RUN_EVALUATION_LOG_DIR
from swebench.harness.docker_build import DEFAULT_DOCKER_TIMEOUT
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import load_swebench_dataset


DEFAULT_HOST = os.environ.get("SWEBENCH_EVAL_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("SWEBENCH_EVAL_PORT", "8080"))
DEFAULT_TIMEOUT = int(os.environ.get("SWEBENCH_EVAL_TIMEOUT", "1800"))
DEFAULT_QUEUE_SIZE = int(os.environ.get("SWEBENCH_EVAL_QUEUE_SIZE", "10000"))
DEFAULT_MODEL_NAME = "evaluation_api"


@dataclass
class Job:
    job_id: str
    instance_id: str
    model_patch: str
    model_name_or_path: str
    timeout: int
    dedupe_key: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    log_dir: str | None = None


class EvaluationService:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        max_workers: int,
        remote_image_namespace: str,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_workers = max_workers
        self.remote_image_namespace = remote_image_namespace
        self.queue_size = queue_size

        dataset = load_swebench_dataset(dataset_name, split)
        self.dataset_by_id = {datum[KEY_INSTANCE_ID]: datum for datum in dataset}

        self._test_spec_cache: dict[str, TestSpec] = {}
        self._jobs: dict[str, Job] = {}
        self._dedupe_to_job: dict[str, str] = {}
        self._queue: Queue[str] = Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._client = docker.from_env(timeout=DEFAULT_DOCKER_TIMEOUT)

    def start(self) -> None:
        for idx in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"swebench-eval-worker-{idx}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def get_test_spec(self, instance_id: str) -> TestSpec:
        with self._lock:
            cached = self._test_spec_cache.get(instance_id)
            if cached is not None:
                return cached
            datum = self.dataset_by_id[instance_id]
            test_spec = make_test_spec(datum)
            self._test_spec_cache[instance_id] = test_spec
            return test_spec

    def submit(
        self,
        instance_id: str,
        model_patch: str,
        model_name_or_path: str,
        timeout: int,
    ) -> tuple[Job, bool]:
        if instance_id not in self.dataset_by_id:
            raise KeyError(instance_id)
        dedupe_key = self._make_dedupe_key(instance_id, model_patch)

        with self._lock:
            existing_job_id = self._dedupe_to_job.get(dedupe_key)
            if existing_job_id is not None:
                existing_job = self._jobs[existing_job_id]
                if existing_job.status != "failed":
                    return existing_job, False

            job = Job(
                job_id=uuid.uuid4().hex,
                instance_id=instance_id,
                model_patch=model_patch,
                model_name_or_path=model_name_or_path,
                timeout=timeout,
                dedupe_key=dedupe_key,
            )
            self._jobs[job.job_id] = job
            self._dedupe_to_job[dedupe_key] = job.job_id

        try:
            self._queue.put_nowait(job.job_id)
        except Full:
            with self._lock:
                self._jobs.pop(job.job_id, None)
                if self._dedupe_to_job.get(dedupe_key) == job.job_id:
                    self._dedupe_to_job.pop(dedupe_key, None)
            raise
        return job, True

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return {
                "dataset_name": self.dataset_name,
                "split": self.split,
                "dataset_size": len(self.dataset_by_id),
                "max_workers": self.max_workers,
                "remote_image_namespace": self.remote_image_namespace,
                "queue_size": self.queue_size,
                "queue_depth": self._queue.qsize(),
                "jobs": counts,
            }

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
            job.log_dir = str(self._log_dir_for_job(job))

        try:
            test_spec = self.get_test_spec(job.instance_id)
            prediction = {
                KEY_INSTANCE_ID: job.instance_id,
                "model_patch": job.model_patch,
                "model_name_or_path": job.model_name_or_path,
            }
            output = run_instance(
                test_spec=test_spec,
                pred=prediction,
                rm_image=False,
                force_rebuild=False,
                client=self._client,
                run_id=job.job_id,
                use_remote_instance_image=True,
                remote_instance_image_namespace=self.remote_image_namespace,
                timeout=job.timeout,
            )
            if output is None:
                raise RuntimeError("Evaluation failed. Check the job log directory.")

            _, report = output
            result = self._summarize_report(job, report)
            with self._lock:
                job.status = "completed"
                job.finished_at = time.time()
                job.result = result
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(exc)

    def _summarize_report(self, job: Job, report: dict[str, Any]) -> dict[str, Any]:
        instance_report = report.get(job.instance_id, {})
        tests_status = instance_report.get("tests_status", {})
        passed_tests = list(tests_status.get("PASS", []))
        failed_tests = list(tests_status.get("FAIL", []))
        return {
            "instance_id": job.instance_id,
            "resolved": bool(instance_report.get("resolved", False)),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "passed_count": len(passed_tests),
            "failed_count": len(failed_tests),
        }

    def _log_dir_for_job(self, job: Job) -> Path:
        return (
            RUN_EVALUATION_LOG_DIR
            / job.job_id
            / job.model_name_or_path.replace("/", "__")
            / job.instance_id
        )

    @staticmethod
    def _make_dedupe_key(instance_id: str, model_patch: str) -> str:
        patch_hash = hashlib.sha256((model_patch or "").encode("utf-8")).hexdigest()
        return f"{instance_id}:{patch_hash}"


class EvaluationHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: EvaluationService):
        self.state = state
        super().__init__(server_address, EvaluationRequestHandler)


class EvaluationRequestHandler(BaseHTTPRequestHandler):
    server: EvaluationHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok", **self.server.state.snapshot()})
            return

        if path.startswith("/jobs/"):
            job_id = path.split("/")[-1]
            job = self.server.state.get_job(job_id)
            if job is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown job_id"})
                return
            self._write_json(HTTPStatus.OK, self._job_payload(job))
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path != "/evaluate":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
            return

        body = self._read_json_body()
        if body is None:
            return

        instance_id = body.get("instance_id")
        model_patch = body.get("model_patch")
        model_name_or_path = body.get("model_name_or_path", DEFAULT_MODEL_NAME)
        timeout = body.get("timeout", DEFAULT_TIMEOUT)

        if not isinstance(instance_id, str) or not instance_id:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "instance_id must be a non-empty string"})
            return
        if not isinstance(model_patch, str):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "model_patch must be a string"})
            return
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "model_name_or_path must be a non-empty string"})
            return
        if not isinstance(timeout, int) or timeout <= 0:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "timeout must be a positive integer"})
            return

        try:
            job, created = self.server.state.submit(
                instance_id=instance_id,
                model_patch=model_patch,
                model_name_or_path=model_name_or_path,
                timeout=timeout,
            )
        except KeyError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": f"Unknown instance_id: {instance_id}"})
            return
        except Full:
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Evaluation queue is full. Retry later."},
            )
            return

        status = HTTPStatus.ACCEPTED if created else HTTPStatus.OK
        self._write_json(status, self._job_payload(job))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any] | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing Content-Length header"})
            return None
        try:
            raw = self.rfile.read(int(content_length))
            return json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
            return None

    def _job_payload(self, job: Job) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "instance_id": job.instance_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
        if job.result is not None:
            payload["result"] = job.result
        if job.error is not None:
            payload["error"] = job.error
        if job.status == "failed" and job.log_dir is not None:
            payload["log_dir"] = job.log_dir
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(
    dataset_name: str,
    split: str,
    max_workers: int,
    remote_image_namespace: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    service = EvaluationService(
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        remote_image_namespace=remote_image_namespace,
    )
    service.start()

    server = EvaluationHTTPServer((host, port), service)
    print(
        json.dumps(
            {
                "status": "starting",
                "host": host,
                "port": port,
                "dataset_name": dataset_name,
                "split": split,
                "dataset_size": len(service.dataset_by_id),
                "max_workers": max_workers,
                "remote_image_namespace": remote_image_namespace,
                "queue_size": service.queue_size,
            }
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or local dataset path")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to serve")
    parser.add_argument("--max_workers", type=int, required=True, help="Maximum concurrent evaluations")
    parser.add_argument(
        "--remote_image_namespace",
        type=str,
        required=True,
        help="Container registry namespace for remote instance images",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind the HTTP server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the HTTP server")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        split=args.split,
        max_workers=args.max_workers,
        remote_image_namespace=args.remote_image_namespace,
        host=args.host,
        port=args.port,
    )
