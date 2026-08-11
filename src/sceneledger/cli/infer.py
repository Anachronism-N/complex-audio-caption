"""``sceneledger-infer`` CLI: run an audio captioner over a TAC-mini manifest.

Writes one prediction Ledger per line (the canonical JSON form that
:mod:`sceneledger.cli.evaluate` consumes) plus a parse report per sample.

::

    python -m sceneledger.cli.infer \
      --manifest data/derived/tac_mini/manifest.jsonl \
      --audio-base /tmp/tac_mini \
      --backend mock \
      --output reports/b0_predictions.jsonl \
      --report reports/b0_infer_report.json

``--backend mock`` uses :class:`MockMossAdapter` (no model needed).
``--backend moss`` uses :class:`MossAdapter` (requires the moss-audio env).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.eval.parser import ParseReport, parse_model_output
from sceneledger.models.moss_adapter import (
    MockMossAdapter,
    MockMossAdapterConfig,
    MossAdapter,
    MossAdapterConfig,
)
from sceneledger.models.target_formatter import atomic_to_ledger, canonical_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-infer")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--audio-base",
        default=".",
        help="root dir that manifest mixture_path values are relative to.",
    )
    parser.add_argument("--backend", choices=["mock", "moss"], default="mock")
    parser.add_argument("--model-path", default=None, help="local weights dir (moss backend)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--lora-path", default=None, help="path to trained LoRA adapter (B1/B2)")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding (faster, deterministic)")
    parser.add_argument("--output", required=True, help="predictions JSONL path")
    parser.add_argument("--report", default=None, help="parse report JSON path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--style", default="brief")
    parser.add_argument("--include-lyrics", action="store_true")
    parser.add_argument(
        "--split-contract",
        default=None,
        help="passed split_contract.json; required together with --expected-split",
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "val", "test"),
        default=None,
        help="assert that --manifest is exactly this frozen split",
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
            require_split_manifest,
        )

        require_experiment_data_summary(args.data_gate_summary, args.split_contract)
        split_contract = require_split_manifest(
            args.split_contract, args.expected_split, args.manifest
        )

    entries = read_manifest(args.manifest)
    if args.limit is not None:
        if split_contract is not None:
            parser.error("--limit is forbidden when a frozen split contract is active")
        entries = entries[: args.limit]

    if args.backend == "moss":
        cfg = MossAdapterConfig()
        if args.model_path:
            cfg.model_path = args.model_path
        cfg.device = args.device
        cfg.dtype = args.dtype
        cfg.max_new_tokens = args.max_new_tokens
        if args.lora_path:
            cfg.lora_path = args.lora_path
        cfg.greedy = args.greedy
        adapter = MossAdapter(cfg)
    else:
        adapter = MockMossAdapter(MockMossAdapterConfig())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    n_ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            sid = entry.scene["scene_id"]
            duration = float(entry.scene["duration"])
            audio_path = str(Path(args.audio_base) / entry.mixture_path)
            prompt = canonical_prompt(style=args.style, include_lyrics=args.include_lyrics)

            if args.backend == "mock":
                target_ledger = Ledger.model_validate(entry.target_ledger)
                raw_text = adapter.infer_from_ledger(target_ledger, sid)
            else:
                raw_text = adapter.infer(audio_path, prompt, sample_id=sid, duration=duration)

            # parse the (atomic-token or free-form) output into a Ledger
            pred_ledger, report = _parse_output(raw_text, sid, duration)
            f.write(json.dumps(pred_ledger.model_dump(mode="json"), ensure_ascii=False) + "\n")
            reports.append(
                {
                    "sample_id": sid,
                    "ok": report.ok,
                    "strict_format_success": report.strict_format_success,
                    "events_recovered": report.events_recovered,
                    "events_rejected": report.events_rejected,
                    "warnings": report.warnings[:5],
                    "raw_text": raw_text[:500] if args.backend == "moss" else None,
                }
            )
            if report.ok:
                n_ok += 1
            if (i + 1) % 100 == 0:
                print(f"[infer] {i + 1}/{len(entries)}", file=sys.stderr)

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "backend": args.backend,
            "n_samples": len(entries),
            "n_ok": n_ok,
            "strict_format_success_rate": round(
                sum(1 for r in reports if r["strict_format_success"]) / max(1, len(reports)), 4
            ),
            "mean_events_recovered": round(
                sum(r["events_recovered"] for r in reports) / max(1, len(reports)), 3
            ),
            "total_events_rejected": sum(r["events_rejected"] for r in reports),
            "manifest_path": str(Path(args.manifest).resolve()),
            "split_contract_path": (
                str(Path(args.split_contract).resolve()) if args.split_contract else None
            ),
            "data_gate_summary_path": (
                str(Path(args.data_gate_summary).resolve())
                if args.data_gate_summary
                else None
            ),
            "expected_split": args.expected_split,
            "dataset_id": split_contract.get("dataset_id") if split_contract else None,
            "samples": reports,
        }
        rp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[infer] {len(entries)} samples via {args.backend} -> {out_path} "
        f"(strict ok={n_ok})",
        file=sys.stderr,
    )
    return 0


def _parse_output(raw_text: str, sample_id: str, duration: float) -> tuple[Ledger, ParseReport]:
    """Parse model output: try atomic-token first, fall back to tolerant XML parser."""
    # atomic-token path (B2/B0 with time markers)
    try:
        ledger = atomic_to_ledger(raw_text, sample_id, duration)
        if ledger.events:
            report = ParseReport(
                sample_id=sample_id, ok=True, events_recovered=len(ledger.events),
                strict_format_success=True,
            )
            return ledger, report
    except Exception:
        pass
    # tolerant XML / free-form path
    ledger, report = parse_model_output(raw_text, sample_id=sample_id, duration_sec=duration)
    return ledger, report


if __name__ == "__main__":
    raise SystemExit(main())
