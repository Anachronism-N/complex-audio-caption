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
        "--parse-report",
        default=None,
        help="inference parse report JSON; preserves raw format failures",
    )
    parser.add_argument("--tiou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--min-text-similarity",
        type=float,
        default=0.0,
        help="hard token-F1 gate; use 0.1 for the lexical-semantic experiment",
    )
    args = parser.parse_args(argv)

    corpus: CorpusMetrics = evaluate_corpus(
        args.prediction,
        args.reference,
        parse_reports=args.parse_report,
        tiou_threshold=args.tiou_threshold,
        min_text_similarity=args.min_text_similarity,
    )
    payload = corpus.to_dict()
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
