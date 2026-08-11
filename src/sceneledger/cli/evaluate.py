"""``sceneledger-evaluate`` CLI.

Mirrors the contract in ``docs/11_development_plan.md``::

    python -m sceneledger.cli.evaluate \
      --prediction tests/fixtures/predictions.jsonl \
      --reference tests/fixtures/references.jsonl \
      --output reports/p0_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sceneledger.eval.metrics import CorpusMetrics, evaluate_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sceneledger-evaluate",
        description="Evaluate predicted SceneLedger ledgers against references.",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Path to predictions JSONL (one Ledger per line).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to references JSONL (one Ledger per line).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write metrics JSON here. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--split-contract",
        default=None,
        help="passed split_contract.json; required together with --expected-split",
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "val", "test"),
        default=None,
        help="require prediction and reference IDs to equal this frozen split",
    )
    parser.add_argument(
        "--data-gate-summary",
        default=None,
        help="passed experiment_data_summary.json for the same split contract",
    )
    args = parser.parse_args(argv)

    if bool(args.split_contract) != bool(args.expected_split):
        parser.error("--split-contract and --expected-split must be provided together")
    if args.split_contract and not args.data_gate_summary:
        parser.error("--data-gate-summary is required with --split-contract")
    if args.data_gate_summary and not args.split_contract:
        parser.error("--data-gate-summary requires --split-contract")
    split_contract = None
    if args.split_contract:
        from sceneledger.data.experiment_data import (
            require_experiment_data_summary,
            require_ledger_split,
        )

        require_experiment_data_summary(args.data_gate_summary, args.split_contract)
        split_contract = require_ledger_split(
            args.split_contract,
            args.expected_split,
            args.reference,
            role="reference",
        )
        require_ledger_split(
            args.split_contract,
            args.expected_split,
            args.prediction,
            role="prediction",
        )

    corpus: CorpusMetrics = evaluate_corpus(args.prediction, args.reference)
    payload = corpus.to_dict()
    if split_contract is not None:
        payload["experiment_contract"] = {
            "dataset_id": split_contract["dataset_id"],
            "split": args.expected_split,
            "split_contract_path": str(Path(args.split_contract).resolve()),
            "data_gate_summary_path": str(Path(args.data_gate_summary).resolve()),
        }
    text = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        # concise summary to stderr/stdout
        print(
            f"[sceneledger] {corpus.n_samples} samples | "
            f"event-F1={corpus.macro_event_f1:.3f} | "
            f"SegF1@100ms={corpus.macro_seg_f1_100ms:.3f} | "
            f"onset-MAE={corpus.mean_onset_mae:.3f}s | "
            f"halluc={corpus.total_hallucination} omit={corpus.total_omission} | "
            f"-> {out}",
            file=sys.stderr,
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
