import json

import docker
import pytest

from swebench.harness import eval_server, swegym_local
from swebench.harness.constants import KEY_INSTANCE_ID


class FakeImages:
    def __init__(self, available):
        self.available = set(available)
        self.lookups = []

    def get(self, image_name):
        self.lookups.append(image_name)
        if image_name not in self.available:
            raise docker.errors.ImageNotFound(image_name)
        return object()


class FakeDockerClient:
    def __init__(self, available):
        self.images = FakeImages(available)
        self.api = object()


class FakeTestSpec:
    instance_id = "repo__pkg-1"
    instance_image_key = "sweb.eval.x86_64.repo__pkg-1:latest"


def make_service(monkeypatch, tmp_path, datum, available, image_map_path=None):
    fake_client = FakeDockerClient(available)
    monkeypatch.setattr(
        eval_server,
        "load_swebench_dataset",
        lambda dataset_name, split: [datum],
    )
    monkeypatch.setattr(eval_server.docker, "from_env", lambda timeout: fake_client)

    service = swegym_local.LocalEvaluationService(
        dataset_name="dataset",
        split="train",
        max_workers=1,
        remote_image_namespace="docker.io/example",
        log_dir=tmp_path,
        image_map_path=image_map_path,
    )
    service._test_spec_cache[datum[KEY_INSTANCE_ID]] = FakeTestSpec()
    return service, fake_client


def test_local_service_resolves_metadata_docker_image(monkeypatch, tmp_path):
    image_name = "swebench/swesmith.x86_64.repo.commit:latest"
    datum = {
        KEY_INSTANCE_ID: "repo__pkg-1",
        "image_name": image_name,
    }
    service, fake_client = make_service(monkeypatch, tmp_path, datum, {image_name})

    assert service.resolve_local_image("repo__pkg-1") == image_name
    assert fake_client.images.lookups == [image_name]


def test_local_service_rejects_missing_docker_image(monkeypatch, tmp_path):
    datum = {
        KEY_INSTANCE_ID: "repo__pkg-1",
        "image_name": "swebench/swesmith.x86_64.repo.commit:latest",
    }
    service, _ = make_service(monkeypatch, tmp_path, datum, set())

    with pytest.raises(swegym_local.LocalImageUnavailable, match="Images not downloaded"):
        service.submit(
            instance_id="repo__pkg-1",
            model_patch="",
            model_name_or_path="model",
            timeout=1,
        )

    assert service.snapshot()["jobs"]["queued"] == 0


def test_local_service_runs_with_resolved_local_image(monkeypatch, tmp_path):
    image_name = "swebench/swesmith.x86_64.repo.commit:latest"
    datum = {
        KEY_INSTANCE_ID: "repo__pkg-1",
        "image_name": image_name,
    }
    service, _ = make_service(monkeypatch, tmp_path, datum, {image_name})
    job = eval_server.Job(
        job_id="job-1",
        instance_id="repo__pkg-1",
        model_patch="",
        model_name_or_path="model",
        timeout=1,
        dedupe_key="key",
    )
    service._jobs[job.job_id] = job
    captured = {}

    def fake_run_instance(**kwargs):
        captured["image"] = kwargs["test_spec"].instance_image_key
        with pytest.raises(swegym_local.LocalImageUnavailable):
            kwargs["client"].api.pull("remote")
        return (
            "repo__pkg-1",
            {
                "repo__pkg-1": {
                    "resolved": True,
                    "tests_status": {"PASS": ["test_ok"], "FAIL": []},
                }
            },
        )

    monkeypatch.setattr(swegym_local, "run_instance", fake_run_instance)

    service._run_job(job.job_id)

    assert captured["image"] == image_name
    assert job.status == "completed"
    assert job.result["resolved"] is True


def test_local_service_runs_with_resolved_apptainer_image(monkeypatch, tmp_path):
    image = tmp_path / "images" / "repo.sif"
    image.parent.mkdir()
    image.write_text("")
    image_map_path = tmp_path / "image_map.json"
    image_map_path.write_text(
        json.dumps(
            {
                "repo__pkg-1": {
                    "status": "completed",
                    "local_image": str(image),
                    "image_format": "sif",
                }
            }
        )
    )
    datum = {KEY_INSTANCE_ID: "repo__pkg-1"}
    service, _ = make_service(
        monkeypatch,
        tmp_path,
        datum,
        available=set(),
        image_map_path=image_map_path,
    )
    job = eval_server.Job(
        job_id="job-1",
        instance_id="repo__pkg-1",
        model_patch="",
        model_name_or_path="model",
        timeout=1,
        dedupe_key="key",
    )
    service._jobs[job.job_id] = job
    captured = {}

    def fake_run_apptainer_instance(**kwargs):
        captured["image"] = kwargs["image"]
        return (
            "repo__pkg-1",
            {
                "repo__pkg-1": {
                    "resolved": True,
                    "tests_status": {"PASS": ["test_ok"], "FAIL": []},
                }
            },
        )

    monkeypatch.setattr(swegym_local, "run_apptainer_instance", fake_run_apptainer_instance)

    service._run_job(job.job_id)

    assert service.resolve_local_runtime_image("repo__pkg-1").runtime == "apptainer"
    assert captured["image"] == str(image)
    assert job.status == "completed"
    assert job.result["resolved"] is True


def test_resolve_local_apptainer_image_from_map(tmp_path):
    image = tmp_path / "images" / "repo.sif"
    image.parent.mkdir()
    image.write_text("")
    image_map_path = tmp_path / "image_map.json"
    image_map_path.write_text(
        json.dumps(
            {
                "repo__pkg-1": {
                    "status": "completed",
                    "local_image": str(image),
                    "image_format": "sif",
                }
            }
        )
    )

    assert (
        swegym_local.resolve_local_apptainer_image("repo__pkg-1", image_map_path)
        == str(image)
    )


def test_resolve_local_apptainer_image_rejects_incomplete_record(tmp_path):
    image_map_path = tmp_path / "image_map.json"
    image_map_path.write_text(
        json.dumps({"repo__pkg-1": {"status": "failed", "local_image": "missing.sif"}})
    )

    with pytest.raises(FileNotFoundError, match="No completed local image"):
        swegym_local.resolve_local_apptainer_image("repo__pkg-1", image_map_path)
