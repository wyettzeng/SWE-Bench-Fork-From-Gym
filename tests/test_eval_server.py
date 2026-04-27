import json

from swebench.harness import eval_server
from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.run_evaluation import run_instance


def test_eval_server_uses_configured_log_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eval_server,
        "load_swebench_dataset",
        lambda dataset_name, split: [{KEY_INSTANCE_ID: "repo__pkg-1"}],
    )
    monkeypatch.setattr(eval_server.docker, "from_env", lambda timeout: object())

    service = eval_server.EvaluationService(
        dataset_name="dataset",
        split="test",
        max_workers=1,
        remote_image_namespace="namespace",
        log_dir=tmp_path,
    )
    job = eval_server.Job(
        job_id="job-1",
        instance_id="repo__pkg-1",
        model_patch="patch",
        model_name_or_path="org/model",
        timeout=1,
        dedupe_key="key",
    )

    assert service.snapshot()["log_dir"] == str(tmp_path)
    assert service._log_dir_for_job(job) == (
        tmp_path / "job-1" / "org__model" / "repo__pkg-1"
    )


def test_run_instance_reads_existing_report_from_custom_log_dir(tmp_path):
    class TestSpec:
        instance_id = "repo__pkg-1"
        instance_image_key = "repo:pkg-1"

    report = {"repo__pkg-1": {"resolved": True}}
    report_path = tmp_path / "run-1" / "model" / "repo__pkg-1" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report))

    assert run_instance(
        test_spec=TestSpec(),
        pred={
            KEY_INSTANCE_ID: "repo__pkg-1",
            "model_patch": "",
            "model_name_or_path": "model",
        },
        rm_image=False,
        force_rebuild=False,
        client=object(),
        run_id="run-1",
        log_dir_root=tmp_path,
        quiet=True,
    ) == ("repo__pkg-1", report)
