"""Call a local or hosted chat-completions-compatible LLM for recipe tasks.

The API key is read only from the named environment variable and is never
written to output.  The JSONL output checkpoints after every task and can be
resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
        rows.append(payload)
    return rows


def _request(
    *,
    endpoint: str,
    model: str,
    task: dict[str, object],
    api_key: str | None,
    timeout_sec: float,
    temperature: float,
    json_mode: bool,
) -> str:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_prompt"]},
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected LLM response structure: {body}") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty content")
    return content.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True, help="full /chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--no-auth", action="store_true", help="for a trusted local endpoint")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    if not args.endpoint.startswith(("http://", "https://")):
        parser.error("--endpoint must be an HTTP(S) URL")
    api_key = None if args.no_auth else os.environ.get(args.api_key_env)
    if not args.no_auth and not api_key:
        parser.error(f"environment variable {args.api_key_env!r} is not set")

    tasks = _read_jsonl(Path(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_jsonl(output) if output.is_file() else []
    completed = {str(row["task_id"]) for row in existing_rows}
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks, 1):
            task_id = str(task["task_id"])
            if task_id in completed:
                continue
            last_error: Exception | None = None
            for attempt in range(args.max_retries):
                try:
                    content = _request(
                        endpoint=args.endpoint,
                        model=args.model,
                        task=task,
                        api_key=api_key,
                        timeout_sec=args.timeout_sec,
                        temperature=args.temperature,
                        json_mode=args.json_mode,
                    )
                    handle.write(
                        json.dumps(
                            {"task_id": task_id, "response": content},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    print(f"completed {index}/{len(tasks)} task_id={task_id}", flush=True)
                    last_error = None
                    break
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 < args.max_retries:
                        time.sleep(min(30.0, 2.0 ** attempt))
            if last_error is not None:
                raise RuntimeError(f"LLM task failed after retries: {task_id}") from last_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
