# Evaluation Server

## Use Case

`swebench.harness.eval_server` exposes the SWE-bench evaluation harness as an on-demand HTTP service.

This is intended for workflows where generation and evaluation are decoupled, for example:

- RL training, where a trainer generates a patch and immediately sends it for evaluation
- Online evaluation of model outputs without first writing a full batch predictions file
- Multi-client setups where many workers submit patches to a shared evaluator

The server:

- loads one dataset and split at startup
- keeps an in-memory map from `instance_id` to dataset row
- accepts evaluation requests one at a time over HTTP
- queues requests and processes up to `max_workers` concurrently
- always uses remote instance images
- pulls remote images on demand only when a job actually runs

This server does not pre-pull images and does not save a separate summary file. Job state is kept in memory for the life of the process, while detailed harness logs are still written under `logs/run_evaluation/`.

## Start The Server

```bash
python -m swebench.harness.eval_server \
  --dataset_name SWE-Gym/SWE-Gym \
  --split train \
  --max_workers 12 \
  --remote_image_namespace docker.io/xingyaoww
```

Arguments:

- `--dataset_name`: Hugging Face dataset name or local dataset path
- `--split`: dataset split to serve
- `--max_workers`: maximum number of concurrent evaluations
- `--remote_image_namespace`: registry namespace for remote instance images
- `--host`: optional, default `0.0.0.0`
- `--port`: optional, default `8080`

Example health check:

```bash
curl http://127.0.0.1:8080/health
```

## Endpoints

### `GET /health`

Returns a lightweight snapshot of server state.

Expected input:

- no request body

Example:

```bash
curl http://127.0.0.1:8080/health
```

Example output:

```json
{
  "status": "ok",
  "dataset_name": "SWE-Gym/SWE-Gym",
  "split": "train",
  "dataset_size": 12345,
  "max_workers": 12,
  "remote_image_namespace": "docker.io/xingyaoww",
  "queue_size": 10000,
  "queue_depth": 3,
  "jobs": {
    "queued": 3,
    "running": 12,
    "completed": 25,
    "failed": 1
  }
}
```

Field meanings:

- `status`: simple liveness indicator
- `dataset_name`: dataset configured at startup
- `split`: split configured at startup
- `dataset_size`: number of dataset rows loaded into memory
- `max_workers`: concurrency limit
- `remote_image_namespace`: remote image registry namespace in use
- `queue_size`: maximum queue capacity
- `queue_depth`: current number of queued, not-yet-running jobs
- `jobs`: counts of jobs by status since process start

### `POST /evaluate`

Submits one evaluation request.

Expected input:

- method: `POST`
- content type: JSON
- body fields:

`instance_id`

- type: `string`
- required: yes
- meaning: SWE-bench instance id already present in the loaded dataset/split

`model_patch`

- type: `string`
- required: yes
- meaning: unified diff patch to apply and evaluate

`model_name_or_path`

- type: `string`
- required: no
- default: `"evaluation_api"`
- meaning: label used in harness logs

`timeout`

- type: `integer`
- required: no
- default: `1800`
- meaning: max evaluation runtime in seconds for this job

Example request:

```bash
curl -X POST http://127.0.0.1:8080/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "instance_id": "pandas-dev__pandas-12345",
    "model_patch": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
    "model_name_or_path": "qwen-test",
    "timeout": 1800
  }'
```

Successful response:

- status code: `202 Accepted` for a new job
- status code: `200 OK` if an identical `(instance_id, model_patch)` request is already known and is reused

Example output for a newly queued job:

```json
{
  "job_id": "2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30",
  "instance_id": "pandas-dev__pandas-12345",
  "status": "queued",
  "created_at": 1775948123.12,
  "started_at": null,
  "finished_at": null
}
```

Possible error responses:

- `400 Bad Request`
  - invalid or missing JSON
  - unknown `instance_id`
  - non-string `model_patch`
  - invalid `timeout`
- `503 Service Unavailable`
  - server queue is full

Notes:

- The server deduplicates by `(instance_id, model_patch)`.
- If the same request is submitted again while queued, running, or completed, the same job is returned.
- Failed jobs are retryable by resubmitting the same request.

### `GET /jobs/<job_id>`

Returns the current status of one job.

Expected input:

- method: `GET`
- path parameter: `job_id`
- no request body

Example:

```bash
curl http://127.0.0.1:8080/jobs/2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30
```

## Job Output Schema

The response always includes:

- `job_id`: unique job id string
- `instance_id`: instance being evaluated
- `status`: one of `queued`, `running`, `completed`, `failed`
- `created_at`: Unix timestamp
- `started_at`: Unix timestamp or `null`
- `finished_at`: Unix timestamp or `null`

If `status == "completed"`, the response also includes:

- `result.instance_id`: evaluated instance id
- `result.resolved`: boolean
- `result.passed_tests`: list of test names that passed
- `result.failed_tests`: list of test names that failed
- `result.passed_count`: integer
- `result.failed_count`: integer

Example completed response:

```json
{
  "job_id": "2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30",
  "instance_id": "pandas-dev__pandas-12345",
  "status": "completed",
  "created_at": 1775948123.12,
  "started_at": 1775948124.02,
  "finished_at": 1775948298.41,
  "result": {
    "instance_id": "pandas-dev__pandas-12345",
    "resolved": false,
    "passed_tests": [
      "pandas.tests.test_example::test_a"
    ],
    "failed_tests": [
      "pandas.tests.test_example::test_b"
    ],
    "passed_count": 1,
    "failed_count": 1
  }
}
```

If `status == "failed"`, the response also includes:

- `error`: short error string
- `log_dir`: path to the harness log directory when available

Example failed response:

```json
{
  "job_id": "2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30",
  "instance_id": "pandas-dev__pandas-12345",
  "status": "failed",
  "created_at": 1775948123.12,
  "started_at": 1775948124.02,
  "finished_at": 1775948130.55,
  "error": "Evaluation failed. Check the job log directory.",
  "log_dir": "logs/run_evaluation/2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30/qwen-test/pandas-dev__pandas-12345"
}
```

## Operational Notes

- The server keeps job state in memory only. Restarting the server loses the in-memory job registry.
- Detailed harness artifacts are still written to `logs/run_evaluation/<job_id>/<model_name_or_path>/<instance_id>/`.
- Remote instance images are pulled lazily when a worker actually starts a job.
- The server does not expose cancellation, deletion, or persistence APIs.
- The server assumes the underlying machine can run Docker-based SWE-bench evaluations.
