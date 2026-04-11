#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("logs/run_evaluation/gym_pandas_qwen3/hosted_vllm__Qwen__Qwen3-Coder-30B-A3B-Instruct")

DIFF_STARTERS = ("diff --git ", "--- ", "*** ")


def classify_patch(text: str) -> tuple[str, str]:
    stripped = text.lstrip()
    first_line = stripped.splitlines()[0] if stripped else ""

    if not stripped:
        return "empty", first_line

    if stripped.startswith("diff --git "):
        if "&& echo" in text or "=== PATCH CONTENT ===" in text:
            return "diff_with_extraneous_wrapper", first_line
        if re.search(r"^@@ [^@]*,\d+,\d+ @@", text, flags=re.M):
            return "invalid_unified_hunk_header", first_line
        return "starts_as_diff", first_line

    if any(stripped.startswith(prefix) for prefix in DIFF_STARTERS):
        return "other_patch_like", first_line

    if first_line.startswith("```") or "```" in text:
        return "fenced_or_formatted_text", first_line

    if first_line.startswith("===") or "PATCH CONTENT" in first_line:
        return "wrapper_text", first_line

    if first_line.startswith('"""') or first_line.startswith("from ") or first_line.startswith("def "):
        return "source_code_instead_of_diff", first_line

    return "plain_text_instead_of_diff", first_line


def main() -> None:
    rows = []
    counts = Counter()
    garbage_ids = []

    for patch_path in sorted(ROOT.glob("*/patch.diff")):
        text = patch_path.read_text(errors="replace")
        category, first_line = classify_patch(text)
        counts[category] += 1

        log_path = patch_path.with_name("run_instance.log")
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        patch_garbage = "Only garbage was found in the patch input." in log_text
        if patch_garbage:
            garbage_ids.append(patch_path.parent.name)

        rows.append(
            {
                "instance": patch_path.parent.name,
                "category": category,
                "first_line": first_line,
                "has_garbage_error": patch_garbage,
            }
        )

    result = {
        "root": str(ROOT),
        "total_patches": len(rows),
        "category_counts": counts,
        "garbage_error_count": len(garbage_ids),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
