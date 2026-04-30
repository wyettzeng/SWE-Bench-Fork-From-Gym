"""Local-only evaluation server for predownloaded SWE-Gym images."""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from queue import Full
from typing import Any
from urllib.parse import urlparse

import docker

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.docker_build import (
    DEFAULT_INSTANCE_IMAGE_NAMESPACE,
    get_remote_instance_image_name,
)
from swebench.harness.eval_server import (
    DEFAULT_HOST,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_PORT,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_TIMEOUT,
    EvaluationDashboard,
    EvaluationRequestHandler,
    EvaluationService,
    Job,
)
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec import TestSpec


IMAGE_METADATA_KEYS = (
    "image",
    "image_name",
    "instance_image",
    "instance_image_name",
    "container_image",
    "execution_image",
    "docker_image",
    "docker_image_name",
    "docker_uri",
    "source_image",
)


class LocalImageUnavailable(RuntimeError):
    def __init__(self, instance_id: str, candidates: list[str]):
        checked = ", ".join(candidates) if candidates else "no Docker image names"
        super().__init__(
            f"Images not downloaded for {instance_id}. "
            f"Checked local Docker cache for: {checked}."
        )
        self.instance_id = instance_id
        self.candidates = candidates


class LocalDockerTestSpec:
    """
    Delegates to a normal TestSpec while forcing the container image tag.

    The standard harness builds or pulls `test_spec.instance_image_key`. For this
    local-only server, the image may already exist under the original registry tag
    from dataset metadata, so the container should be created from that tag
    directly.
    """

    def __init__(self, test_spec: TestSpec, image_name: str):
        self._test_spec = test_spec
        self._image_name = image_name

    @property
    def instance_image_key(self) -> str:
        return self._image_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._test_spec, name)


class LocalOnlyDockerAPI:
    def __init__(self, api: Any, instance_id: str, candidates: list[str]):
        self._api = api
        self._instance_id = instance_id
        self._candidates = candidates

    def pull(self, *args: Any, **kwargs: Any) -> Any:
        raise LocalImageUnavailable(self._instance_id, self._candidates)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


class LocalOnlyDockerClient:
    def __init__(
        self,
        client: docker.DockerClient,
        instance_id: str,
        candidates: list[str],
    ):
        self._client = client
        self.api = LocalOnlyDockerAPI(client.api, instance_id, candidates)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def resolve_local_apptainer_image(instance_id: str, image_map_path: Path | str) -> str:
    image_map = json.loads(Path(image_map_path).read_text())
    record = image_map.get(instance_id)
    if not record or record.get("status") != "completed":
        raise FileNotFoundError(f"No completed local image for {instance_id}")

    local_image = Path(record["local_image"])
    image_format = record.get("image_format", "sif")
    exists = local_image.is_dir() if image_format == "sandbox" else local_image.is_file()
    if not exists:
        raise FileNotFoundError(f"Local image path is missing: {local_image}")

    return str(local_image)


def docker_image_candidates(
    datum: dict[str, Any],
    test_spec: TestSpec,
    remote_image_namespace: str,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        image_name = normalize_docker_image_name(value)
        if image_name is not None and image_name not in seen:
            seen.add(image_name)
            candidates.append(image_name)

    for key in IMAGE_METADATA_KEYS:
        add(datum.get(key))

    for key in ("metadata", "runtime", "container", "images"):
        value = datum.get(key)
        if isinstance(value, dict):
            for image_key in IMAGE_METADATA_KEYS:
                add(value.get(image_key))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for image_key in IMAGE_METADATA_KEYS:
                        add(item.get(image_key))
                else:
                    add(item)

    add(get_remote_instance_image_name(test_spec, remote_image_namespace))
    add(test_spec.instance_image_key)
    return candidates


def normalize_docker_image_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    image_name = value.strip()
    if not image_name:
        return None
    if image_name.startswith("docker://"):
        image_name = image_name[len("docker://") :]
    if image_name.endswith(".sif") or image_name.endswith(".sandbox"):
        return None
    return image_name


class LocalEvaluationService(EvaluationService):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        max_workers: int,
        remote_image_namespace: str = DEFAULT_INSTANCE_IMAGE_NAMESPACE,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        log_dir: Path | str = DEFAULT_LOG_DIR,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            max_workers=max_workers,
            remote_image_namespace=remote_image_namespace,
            queue_size=queue_size,
            log_dir=log_dir,
        )
        self._local_image_cache: dict[str, str] = {}

    def submit(
        self,
        instance_id: str,
        model_patch: str,
        model_name_or_path: str,
        timeout: int,
    ) -> tuple[Job, bool]:
        if instance_id not in self.dataset_by_id:
            raise KeyError(instance_id)
        self.resolve_local_image(instance_id)
        return super().submit(
            instance_id=instance_id,
            model_patch=model_patch,
            model_name_or_path=model_name_or_path,
            timeout=timeout,
        )

    def resolve_local_image(self, instance_id: str) -> str:
        cached = self._local_image_cache.get(instance_id)
        if cached is not None:
            try:
                self._client.images.get(cached)
                return cached
            except docker.errors.ImageNotFound:
                self._local_image_cache.pop(instance_id, None)

        datum = self.dataset_by_id[instance_id]
        test_spec = self.get_test_spec(instance_id)
        candidates = docker_image_candidates(
            datum,
            test_spec,
            self.remote_image_namespace,
        )
        for image_name in candidates:
            try:
                self._client.images.get(image_name)
                self._local_image_cache[instance_id] = image_name
                return image_name
            except docker.errors.ImageNotFound:
                continue
        raise LocalImageUnavailable(instance_id, candidates)

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot["image_mode"] = "local_docker_cache"
        return snapshot

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
            job.log_dir = str(self._log_dir_for_job(job))
        self._notify_status_changed()

        try:
            test_spec = self.get_test_spec(job.instance_id)
            local_image = self.resolve_local_image(job.instance_id)
            local_test_spec = LocalDockerTestSpec(test_spec, local_image)
            prediction = {
                KEY_INSTANCE_ID: job.instance_id,
                "model_patch": job.model_patch,
                "model_name_or_path": job.model_name_or_path,
            }
            output = run_instance(
                test_spec=local_test_spec,
                pred=prediction,
                rm_image=False,
                force_rebuild=False,
                client=LocalOnlyDockerClient(
                    self._client,
                    job.instance_id,
                    [local_image],
                ),
                run_id=job.job_id,
                use_remote_instance_image=True,
                remote_instance_image_namespace=self.remote_image_namespace,
                timeout=job.timeout,
                log_dir_root=self.log_dir,
                quiet=True,
            )
            if output is None:
                try:
                    self._client.images.get(local_image)
                except docker.errors.ImageNotFound as exc:
                    raise LocalImageUnavailable(job.instance_id, [local_image]) from exc
                raise RuntimeError("Evaluation failed. Check the job log directory.")

            _, report = output
            result = self._summarize_report(job, report)
            with self._lock:
                job.status = "completed"
                job.finished_at = time.time()
                job.result = result
            self._notify_status_changed()
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(exc)
            self._notify_status_changed()


class LocalEvaluationRequestHandler(EvaluationRequestHandler):
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
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "instance_id must be a non-empty string"},
            )
            return
        if not isinstance(model_patch, str):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "model_patch must be a string"},
            )
            return
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "model_name_or_path must be a non-empty string"},
            )
            return
        if not isinstance(timeout, int) or timeout <= 0:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "timeout must be a positive integer"},
            )
            return

        try:
            job, created = self.server.state.submit(
                instance_id=instance_id,
                model_patch=model_patch,
                model_name_or_path=model_name_or_path,
                timeout=timeout,
            )
        except KeyError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Unknown instance_id: {instance_id}"},
            )
            return
        except LocalImageUnavailable as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Full:
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Evaluation queue is full. Retry later."},
            )
            return

        status = HTTPStatus.ACCEPTED if created else HTTPStatus.OK
        self._write_json(status, self._job_payload(job))


class LocalEvaluationHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        state: LocalEvaluationService,
    ):
        self.state = state
        super().__init__(server_address, LocalEvaluationRequestHandler)


def main(
    dataset_name: str,
    split: str,
    max_workers: int,
    remote_image_namespace: str = DEFAULT_INSTANCE_IMAGE_NAMESPACE,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_dir: Path | str = DEFAULT_LOG_DIR,
) -> None:
    service = LocalEvaluationService(
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        remote_image_namespace=remote_image_namespace,
        log_dir=log_dir,
    )
    service.start()

    server = LocalEvaluationHTTPServer((host, port), service)
    dashboard = EvaluationDashboard(service, host, port)
    service.set_status_callback(dashboard.refresh)
    dashboard.start()
    try:
        server.serve_forever()
    finally:
        dashboard.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name or local dataset path",
    )
    parser.add_argument("--split", type=str, required=True, help="Dataset split to serve")
    parser.add_argument(
        "--max_workers",
        type=int,
        required=True,
        help="Maximum concurrent evaluations",
    )
    parser.add_argument(
        "--remote_image_namespace",
        type=str,
        default=DEFAULT_INSTANCE_IMAGE_NAMESPACE,
        help=(
            "Registry namespace used only to derive candidate local image tags. "
            "This local server never pulls from the registry."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host to bind the HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind the HTTP server",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-job evaluation logs",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        split=args.split,
        max_workers=args.max_workers,
        remote_image_namespace=args.remote_image_namespace,
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
    )
