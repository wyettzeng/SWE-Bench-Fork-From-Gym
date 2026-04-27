# Evaluation Server

`swebench.harness.eval_server` exposes the SWE-bench harness as a small HTTP
service. The server loads one dataset split at startup, accepts one patch per
request, queues evaluation jobs in memory, and runs up to `--max_workers` jobs
concurrently.

Job state is process-local and is lost when the server exits. Detailed harness
logs are written under the directory configured by `--log_dir`, which defaults
to `logs/run_evaluation/`.

## Starting The Server

```bash
python -m swebench.harness.eval_server \
  --dataset_name SWE-Gym/SWE-Gym \
  --split train \
  --max_workers 12 \
  --remote_image_namespace docker.io/xingyaoww \
  --log_dir /path/to/eval-logs
```

Required arguments:

- `--dataset_name`: Hugging Face dataset name or local dataset path.
- `--split`: Dataset split loaded into memory.
- `--max_workers`: Maximum number of concurrent evaluations.
- `--remote_image_namespace`: Registry namespace for remote instance images.

Optional arguments:

- `--host`: Bind host. Default: `SWEBENCH_EVAL_HOST` or `0.0.0.0`.
- `--port`: Bind port. Default: `SWEBENCH_EVAL_PORT` or `8080`.
- `--log_dir`: Directory for per-job harness logs. Default: `SWEBENCH_EVAL_LOG_DIR` or `logs/run_evaluation`.

Other environment-backed defaults:

- `SWEBENCH_EVAL_TIMEOUT`: Default per-job timeout in seconds. Default: `1800`.
- `SWEBENCH_EVAL_QUEUE_SIZE`: Maximum queued jobs. Default: `10000`.

On startup the process displays a live terminal panel instead of streaming
worker output. The panel shows the bind address, dataset, worker usage, job
counts, log directory, currently running instance IDs, and queued instance IDs.

## HTTP Behavior

All documented responses are JSON with `Content-Type: application/json`.
Trailing slashes are accepted for the documented endpoints.

The request body for `POST /evaluate` must be valid JSON and must include a
`Content-Length` header. Common HTTP clients such as `curl` and `requests`
include this header automatically. The code does not require or validate a
specific `Content-Type` header, but clients should send `application/json`.

There is no authentication, persistence, cancellation endpoint, or streaming
output in this server.

## `GET /health`

Returns a snapshot of the server and queue state.

Expected input:

- Method: `GET`
- Path: `/health`
- Body: none

Example:

```bash
curl http://127.0.0.1:8080/health
```

Output:

```json
{
  "status": "ok",
  "dataset_name": "SWE-Gym/SWE-Gym",
  "split": "train",
  "dataset_size": 12345,
  "max_workers": 12,
  "remote_image_namespace": "docker.io/xingyaoww",
  "log_dir": "/path/to/eval-logs",
  "queue_size": 10000,
  "queue_depth": 3,
  "jobs": {
    "queued": 3,
    "running": 12,
    "completed": 25,
    "failed": 1
  },
  "running_jobs": [],
  "queued_jobs": []
}
```

Output fields:

- `status`: Liveness indicator. Always `"ok"` for this endpoint.
- `dataset_name`: Dataset configured at startup.
- `split`: Split configured at startup.
- `dataset_size`: Number of dataset rows loaded into memory.
- `max_workers`: Concurrent worker limit.
- `remote_image_namespace`: Remote image namespace configured at startup.
- `log_dir`: Root directory where per-job harness logs are written.
- `queue_size`: Maximum queue capacity.
- `queue_depth`: Current number of queued, not-yet-running jobs.
- `jobs`: Counts of in-memory jobs by status since the process started.
- `running_jobs`: Summaries of jobs currently being processed.
- `queued_jobs`: Summaries of jobs waiting for a worker.

## `POST /evaluate`

Submits one patch for evaluation.

Expected input:

- Method: `POST`
- Path: `/evaluate`
- Body: JSON object

Request body fields:

| Field | Required | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `instance_id` | Yes | non-empty string | none | Instance id in the loaded dataset split. |
| `model_patch` | Yes | string | none | Unified diff patch to apply and evaluate. Empty string is accepted. |
| `model_name_or_path` | No | non-empty string | `"evaluation_api"` | Label used when writing harness logs. |
| `timeout` | No | positive integer | `SWEBENCH_EVAL_TIMEOUT` or `1800` | Per-job evaluation timeout in seconds. |

Example:

```bash
curl -X POST http://127.0.0.1:8080/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "instance_id": "pandas-dev__pandas-12345",
  "model_patch": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
  "model_name_or_path": "qwen-test",
  "timeout": 1800
}
JSON
```

Successful output:

- `202 Accepted`: A new job was created and queued.
- `200 OK`: An identical non-failed job already exists and was returned.

The server deduplicates jobs by `instance_id` plus a SHA-256 hash of
`model_patch`. `model_name_or_path` and `timeout` are not part of the dedupe
key. If a matching job has status `queued`, `running`, or `completed`, the same
job is returned. If the matching job failed, submitting the same request creates
a new job.

Example queued response:

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

Validation and queue errors:

| Status | Body | Cause |
| --- | --- | --- |
| `400 Bad Request` | `{"error": "Missing Content-Length header"}` | No `Content-Length` header. |
| `400 Bad Request` | `{"error": "Request body must be valid JSON"}` | Body cannot be parsed as JSON. |
| `400 Bad Request` | `{"error": "instance_id must be a non-empty string"}` | Missing, empty, or non-string `instance_id`. |
| `400 Bad Request` | `{"error": "model_patch must be a string"}` | Missing or non-string `model_patch`. |
| `400 Bad Request` | `{"error": "model_name_or_path must be a non-empty string"}` | Empty or non-string `model_name_or_path`. |
| `400 Bad Request` | `{"error": "timeout must be a positive integer"}` | Missing default is valid, but provided value must be a positive integer. |
| `400 Bad Request` | `{"error": "Unknown instance_id: <instance_id>"}` | `instance_id` is not in the loaded dataset split. |
| `503 Service Unavailable` | `{"error": "Evaluation queue is full. Retry later."}` | The in-memory queue has reached `queue_size`. |

## `GET /jobs/<job_id>`

Returns the current status and final result, if available, for one job.

Expected input:

- Method: `GET`
- Path: `/jobs/<job_id>`
- Body: none

Example:

```bash
curl http://127.0.0.1:8080/jobs/2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30
```

If the job id is unknown, the server returns:

```json
{
  "error": "Unknown job_id"
}
```

with status `404 Not Found`.

## Job Response Schema

`POST /evaluate` and `GET /jobs/<job_id>` both return the same job payload
shape.

Fields always present:

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Unique server-generated job id. |
| `instance_id` | string | Instance being evaluated. |
| `status` | string | One of `queued`, `running`, `completed`, or `failed`. |
| `created_at` | number | Unix timestamp, in seconds, when the job object was created. |
| `started_at` | number or null | Unix timestamp when a worker started the job. |
| `finished_at` | number or null | Unix timestamp when the job completed or failed. |

Queued response:

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

Running response:

```json
{
  "job_id": "2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30",
  "instance_id": "pandas-dev__pandas-12345",
  "status": "running",
  "created_at": 1775948123.12,
  "started_at": 1775948124.02,
  "finished_at": null
}
```

Completed jobs include a `result` object:

| Field | Type | Meaning |
| --- | --- | --- |
| `result.instance_id` | string | Evaluated instance id. |
| `result.resolved` | boolean | Whether the harness report marked the instance as resolved. |
| `result.passed_tests` | string array | Test names from `tests_status.PASS`. |
| `result.failed_tests` | string array | Test names from `tests_status.FAIL`. |
| `result.passed_count` | integer | Length of `passed_tests`. |
| `result.failed_count` | integer | Length of `failed_tests`. |

Example completed response:

```json
{
  "job_id": "2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30",
  "instance_id": "pandas-dev__pandas-12345",
  "status": "completed",
  "created_at": 1775948123.12,
  "started_at": 1775948124.02,
  "finished_at": 1775948298.41,
  "log_dir": "/path/to/eval-logs/2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30/qwen-test/pandas-dev__pandas-12345",
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

Running, completed, and failed jobs include `log_dir` once the job has started
far enough for the server to compute the harness log path. Failed jobs also
include an `error` string.

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
  "log_dir": "/path/to/eval-logs/2d7f1f5b0c6a4b3e9b6bbf4e7b6a4e30/qwen-test/pandas-dev__pandas-12345"
}
```

## Unknown Endpoints

Any unsupported `GET` or `POST` path returns:

```json
{
  "error": "Unknown endpoint"
}
```

with status `404 Not Found`.

## Evaluation Details

Each worker builds a prediction object with:

```json
{
  "instance_id": "<instance_id>",
  "model_patch": "<model_patch>",
  "model_name_or_path": "<model_name_or_path>"
}
```

and passes it to `run_instance()` with these fixed behaviors:

- `run_id` is the server-generated `job_id`.
- `use_remote_instance_image` is always `true`.
- `remote_instance_image_namespace` comes from `--remote_image_namespace`.
- `rm_image` is `false`.
- `force_rebuild` is `false`.

The HTTP result is a summary of the harness report for the requested
`instance_id`, not the full raw report.
