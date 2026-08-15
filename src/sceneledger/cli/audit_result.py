"""CLI for certifying that an evaluation is genuinely held out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sceneledger.eval.result_validity import audit_evaluation_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-audit-result")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--inference-report", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split-contract")
    parser.add_argument("--data-gate-summary")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit code 2 after writing the report when certification fails",
    )
    args = parser.parse_args(argv)

    payload = audit_evaluation_result(
        train_config_path=args.train_config,
        eval_manifest_path=args.eval_manifest,
        metrics_path=args.metrics,
        inference_report_path=args.inference_report,
        repo_root=args.repo_root,
        split_contract_path=args.split_contract,
        data_gate_summary_path=args.data_gate_summary,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[result-audit] status={payload['status']} "
        f"failed={payload['failed_checks']} -> {output}",
        file=sys.stderr,
    )
    return 2 if args.require_pass and payload["pass"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
