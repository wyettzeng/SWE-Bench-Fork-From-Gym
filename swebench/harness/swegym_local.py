"""Local-only evaluation server for predownloaded SWE-Gym images."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from queue import Full
from typing import Any
from urllib.parse import urlparse

import docker

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
)
from swebench.harness.docker_build import (
    DEFAULT_INSTANCE_IMAGE_NAMESPACE,
    close_logger,
    get_remote_instance_image_name,
    setup_logger,
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
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec import TestSpec


DEFAULT_APPTAINER_IMAGE_MAP_ENV_VAR = "SWEGYM_LOCAL_IMAGE_MAP_PATH"
DEFAULT_APPTAINER_IMAGE_ROOT = Path("SWE-Actor") / "swegym_pandas_apptainer_images"
DEFAULT_APPTAINER_EXECUTABLE_ENV_VAR = "SWEGYM_LOCAL_APPTAINER_EXECUTABLE"
APPTAINER_EVAL_MOUNT = Path("/swebench-eval")
APPTAINER_PATCH_FAILED_EXIT_CODE = 79

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


@dataclass(frozen=True)
class LocalImageResolution:
    runtime: str
    image: str
    source: str
    image_map_path: str | None = None


@dataclass(frozen=True)
class ApptainerCommandResult:
    exit_code: int
    output: str
    timed_out: bool
    runtime: float


class LocalImageUnavailable(RuntimeError):
    def __init__(self, instance_id: str, candidates: list[str], detail: str | None = None):
        checked = ", ".join(candidates) if candidates else "no local image candidates"
        if detail:
            checked = f"{checked}. {detail}"
        super().__init__(
            f"Images not downloaded for {instance_id}. "
            f"Checked local images for: {checked}."
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


def get_default_image_map_path() -> Path | None:
    if configured_path := os.getenv(DEFAULT_APPTAINER_IMAGE_MAP_ENV_VAR):
        return Path(configured_path)
    if scratch_disk := os.getenv("SCRATCH_DISK"):
        return Path(scratch_disk) / DEFAULT_APPTAINER_IMAGE_ROOT / "image_map.json"
    return None


def resolve_local_apptainer_image(instance_id: str, image_map_path: Path | str | None) -> str:
    if image_map_path is None:
        raise FileNotFoundError(
            f"No local Apptainer image map configured for {instance_id}. "
            f"Pass --image_map_path or set ${DEFAULT_APPTAINER_IMAGE_MAP_ENV_VAR} "
            "or $SCRATCH_DISK."
        )

    image_map_file = Path(image_map_path)
    if not image_map_file.is_file():
        raise FileNotFoundError(
            f"Local Apptainer image map is missing for {instance_id}: {image_map_file}"
        )

    image_map = json.loads(image_map_file.read_text())
    record = image_map.get(instance_id)
    if not record or record.get("status") != "completed":
        raise FileNotFoundError(f"No completed local image for {instance_id} in {image_map_file}")

    local_image_value = record.get("local_image")
    if not local_image_value:
        raise FileNotFoundError(
            f"Completed local image record for {instance_id} is missing local_image in {image_map_file}"
        )

    local_image = Path(local_image_value)
    image_format = record.get("image_format", "sif")
    exists = local_image.is_dir() if image_format == "sandbox" else local_image.is_file()
    if not exists:
        raise FileNotFoundError(f"Local image path is missing: {local_image}")

    return str(local_image)


def find_apptainer_executable(configured_executable: str | None = None) -> str:
    executable = (
        configured_executable
        or os.getenv(DEFAULT_APPTAINER_EXECUTABLE_ENV_VAR)
        or shutil.which("apptainer")
        or shutil.which("singularity")
        or "apptainer"
    )
    return str(executable)


def _process_output_to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def run_apptainer_command(
    *,
    executable: str,
    image: str,
    log_dir: Path,
    command: str,
    timeout: int | None = None,
) -> ApptainerCommandResult:
    bind_spec = f"{log_dir.absolute()}:{APPTAINER_EVAL_MOUNT}"
    argv = [
        executable,
        "exec",
        "--writable-tmpfs",
        "--bind",
        bind_spec,
        "--pwd",
        "/testbed",
        image,
        "/bin/bash",
        "-lc",
        command,
    ]
    started_at = time.time()
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return ApptainerCommandResult(
            exit_code=completed.returncode,
            output=completed.stdout,
            timed_out=False,
            runtime=time.time() - started_at,
        )
    except subprocess.TimeoutExpired as exc:
        return ApptainerCommandResult(
            exit_code=-signal.SIGTERM,
            output=_process_output_to_text(exc.stdout),
            timed_out=True,
            runtime=time.time() - started_at,
        )


def run_apptainer_instance(
    *,
    test_spec: TestSpec,
    pred: dict[str, Any],
    image: str,
    run_id: str,
    timeout: int | None,
    log_dir_root: Path | str,
    apptainer_executable: str,
    quiet: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = Path(log_dir_root) / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except Exception:
            pass

    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    log_file = log_dir / "run_instance.log"
    logger = setup_logger(instance_id, log_file)
    patch_path = APPTAINER_EVAL_MOUNT / "patch.diff"
    eval_path = APPTAINER_EVAL_MOUNT / "eval.sh"
    driver_path = APPTAINER_EVAL_MOUNT / "apptainer_eval.sh"
    apply_output_path = log_dir / "apply_patch_output.txt"
    apply_exit_code_path = log_dir / "apply_patch_exit_code.txt"
    diff_before_path = log_dir / "git_diff_before.txt"
    diff_after_path = log_dir / "git_diff_after.txt"
    eval_exit_code_path = log_dir / "eval_exit_code.txt"

    def read_log_artifact(path: Path) -> str:
        return path.read_text(errors="replace") if path.exists() else ""

    try:
        logger.info(f"Using local Apptainer image for {instance_id}: {image}")

        patch_file = log_dir / "patch.diff"
        patch_file.write_text(pred["model_patch"] or "")
        logger.info(f"Intermediate patch for {instance_id} written to {patch_file}")

        eval_file = log_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script)
        logger.info(f"Eval script for {instance_id} written to {eval_file}")

        driver_file = log_dir / "apptainer_eval.sh"
        driver_file.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "set +e",
                    "cd /testbed",
                    (
                        "git apply --allow-empty -v "
                        f"{shlex.quote(patch_path.as_posix())} "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / apply_output_path.name).as_posix())} "
                        "2>&1"
                    ),
                    "apply_code=$?",
                    "if [ \"$apply_code\" -ne 0 ]; then",
                    (
                        "  echo 'Failed to apply patch with git apply, "
                        "trying patch fallback...' "
                        f">> {shlex.quote((APPTAINER_EVAL_MOUNT / apply_output_path.name).as_posix())}"
                    ),
                    (
                        "  patch --batch --fuzz=5 -p1 -i "
                        f"{shlex.quote(patch_path.as_posix())} "
                        f">> {shlex.quote((APPTAINER_EVAL_MOUNT / apply_output_path.name).as_posix())} "
                        "2>&1"
                    ),
                    "  apply_code=$?",
                    "fi",
                    (
                        "echo \"$apply_code\" "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / apply_exit_code_path.name).as_posix())}"
                    ),
                    "if [ \"$apply_code\" -ne 0 ]; then",
                    f"  exit {APPTAINER_PATCH_FAILED_EXIT_CODE}",
                    "fi",
                    (
                        "git diff "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / diff_before_path.name).as_posix())} "
                        "2>&1"
                    ),
                    (
                        f"/bin/bash {shlex.quote(eval_path.as_posix())} "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / 'test_output.txt').as_posix())} "
                        "2>&1"
                    ),
                    "eval_code=$?",
                    (
                        "echo \"$eval_code\" "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / eval_exit_code_path.name).as_posix())}"
                    ),
                    (
                        "git diff "
                        f"> {shlex.quote((APPTAINER_EVAL_MOUNT / diff_after_path.name).as_posix())} "
                        "2>&1"
                    ),
                    "exit 0",
                    "",
                ]
            )
        )

        logger.info(f"Apptainer driver for {instance_id} written to {driver_file}")
        eval_result = run_apptainer_command(
            executable=apptainer_executable,
            image=image,
            log_dir=log_dir,
            command=f"/bin/bash {shlex.quote(driver_path.as_posix())}",
            timeout=timeout,
        )
        logger.info(
            f"Apptainer driver exit={eval_result.exit_code}, "
            f"timed_out={eval_result.timed_out}, runtime={eval_result.runtime:_.2f} seconds"
        )

        apply_output = read_log_artifact(apply_output_path)
        if eval_result.exit_code == APPTAINER_PATCH_FAILED_EXIT_CODE:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{apply_output}")
            raise RuntimeError(f"{APPLY_PATCH_FAIL}:\n{apply_output}")
        if eval_result.exit_code != 0 and not eval_result.timed_out:
            raise RuntimeError(
                f"Apptainer evaluation driver failed with exit code {eval_result.exit_code}:\n"
                f"{eval_result.output}"
            )
        logger.info(f"{APPLY_PATCH_PASS}:\n{apply_output}")

        diff_before = read_log_artifact(diff_before_path).strip()
        logger.info(f"Git diff before:\n{diff_before}")

        test_output_path = log_dir / "test_output.txt"
        if not test_output_path.exists():
            test_output_path.write_text(eval_result.output)
        logger.info(f"Test runtime: {eval_result.runtime:_.2f} seconds")
        logger.info(f"Test output for {instance_id} written to {test_output_path}")
        if eval_result.timed_out:
            with test_output_path.open("a") as output_file:
                output_file.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise RuntimeError(f"Test timed out after {timeout} seconds.")

        eval_exit_code = read_log_artifact(eval_exit_code_path).strip()
        if eval_exit_code:
            logger.info(f"Eval script exit code: {eval_exit_code}")

        diff_after = read_log_artifact(diff_after_path).strip()
        logger.info(f"Git diff after:\n{diff_after}")
        if diff_after != diff_before:
            logger.info("Git diff changed after running eval script")

        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        report_path.write_text(json.dumps(report, indent=4))
        return instance_id, report
    except Exception as exc:
        error_msg = (
            f"Error in Apptainer evaluation for {instance_id}: {exc}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
        if not quiet:
            print(error_msg)
    finally:
        close_logger(logger)
    return None


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
        image_map_path: Path | str | None = None,
        apptainer_executable: str | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            max_workers=max_workers,
            remote_image_namespace=remote_image_namespace,
            queue_size=queue_size,
            log_dir=log_dir,
        )
        self.image_map_path = (
            Path(image_map_path) if image_map_path is not None else get_default_image_map_path()
        )
        self.apptainer_executable = find_apptainer_executable(apptainer_executable)
        self._local_image_cache: dict[str, LocalImageResolution] = {}

    def submit(
        self,
        instance_id: str,
        model_patch: str,
        model_name_or_path: str,
        timeout: int,
    ) -> tuple[Job, bool]:
        if instance_id not in self.dataset_by_id:
            raise KeyError(instance_id)
        self.resolve_local_runtime_image(instance_id)
        return super().submit(
            instance_id=instance_id,
            model_patch=model_patch,
            model_name_or_path=model_name_or_path,
            timeout=timeout,
        )

    def resolve_local_image(self, instance_id: str) -> str:
        return self.resolve_local_runtime_image(instance_id).image

    def resolve_local_runtime_image(self, instance_id: str) -> LocalImageResolution:
        cached = self._local_image_cache.get(instance_id)
        if cached is not None:
            if cached.runtime == "docker":
                try:
                    self._client.images.get(cached.image)
                    return cached
                except docker.errors.ImageNotFound:
                    self._local_image_cache.pop(instance_id, None)
            elif Path(cached.image).is_file() or Path(cached.image).is_dir():
                return cached
            else:
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
                resolution = LocalImageResolution(
                    runtime="docker",
                    image=image_name,
                    source="docker_cache",
                )
                self._local_image_cache[instance_id] = resolution
                return resolution
            except docker.errors.ImageNotFound:
                continue

        try:
            apptainer_image = resolve_local_apptainer_image(instance_id, self.image_map_path)
        except FileNotFoundError as exc:
            raise LocalImageUnavailable(instance_id, candidates, detail=str(exc)) from exc

        resolution = LocalImageResolution(
            runtime="apptainer",
            image=apptainer_image,
            source="image_map",
            image_map_path=str(self.image_map_path) if self.image_map_path is not None else None,
        )
        self._local_image_cache[instance_id] = resolution
        return resolution

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot["image_mode"] = "local_docker_cache_or_apptainer_image_map"
        snapshot["apptainer_image_map_path"] = (
            str(self.image_map_path) if self.image_map_path is not None else None
        )
        snapshot["apptainer_executable"] = self.apptainer_executable
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
            local_resolution = self.resolve_local_runtime_image(job.instance_id)
            prediction = {
                KEY_INSTANCE_ID: job.instance_id,
                "model_patch": job.model_patch,
                "model_name_or_path": job.model_name_or_path,
            }
            if local_resolution.runtime == "docker":
                local_test_spec = LocalDockerTestSpec(test_spec, local_resolution.image)
                output = run_instance(
                    test_spec=local_test_spec,
                    pred=prediction,
                    rm_image=False,
                    force_rebuild=False,
                    client=LocalOnlyDockerClient(
                        self._client,
                        job.instance_id,
                        [local_resolution.image],
                    ),
                    run_id=job.job_id,
                    use_remote_instance_image=True,
                    remote_instance_image_namespace=self.remote_image_namespace,
                    timeout=job.timeout,
                    log_dir_root=self.log_dir,
                    quiet=True,
                )
            elif local_resolution.runtime == "apptainer":
                output = run_apptainer_instance(
                    test_spec=test_spec,
                    pred=prediction,
                    image=local_resolution.image,
                    run_id=job.job_id,
                    timeout=job.timeout,
                    log_dir_root=self.log_dir,
                    apptainer_executable=self.apptainer_executable,
                    quiet=True,
                )
            else:
                raise RuntimeError(f"Unsupported local image runtime: {local_resolution.runtime}")

            if output is None:
                if local_resolution.runtime == "docker":
                    try:
                        self._client.images.get(local_resolution.image)
                    except docker.errors.ImageNotFound as exc:
                        raise LocalImageUnavailable(
                            job.instance_id,
                            [local_resolution.image],
                        ) from exc
                elif not Path(local_resolution.image).exists():
                    raise LocalImageUnavailable(job.instance_id, [local_resolution.image])
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
    image_map_path: Path | str | None = None,
    apptainer_executable: str | None = None,
) -> None:
    service = LocalEvaluationService(
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        remote_image_namespace=remote_image_namespace,
        log_dir=log_dir,
        image_map_path=image_map_path,
        apptainer_executable=apptainer_executable,
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
    parser.add_argument(
        "--image_map_path",
        type=Path,
        default=None,
        help=(
            "Path to the local Apptainer image_map.json. Defaults to "
            f"${DEFAULT_APPTAINER_IMAGE_MAP_ENV_VAR}, or "
            f"$SCRATCH_DISK/{DEFAULT_APPTAINER_IMAGE_ROOT}/image_map.json."
        ),
    )
    parser.add_argument(
        "--apptainer_executable",
        type=str,
        default=None,
        help=(
            "Apptainer/Singularity executable to use for local SIF images. "
            f"Defaults to ${DEFAULT_APPTAINER_EXECUTABLE_ENV_VAR}, apptainer, then singularity."
        ),
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
        image_map_path=args.image_map_path,
        apptainer_executable=args.apptainer_executable,
    )
