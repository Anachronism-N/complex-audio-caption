"""Re-evaluate committed raw generations without loading an audio model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.experiment_data import file_sha256
from sceneledger.eval.forensic_replay import replay_raw_inference


def _write_ledgers(path: Path, ledgers: dict) -> None:
    path.write_text(
        "".join(
            json.dumps(ledger.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for ledger in ledgers.values()
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-forensic-replay")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inference-report", required=True)
    parser.add_argument("--original-metrics")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    predictions, references, metrics, report = replay_raw_inference(
        train_config_path=args.train_config,
        manifest_path=args.manifest,
        inference_report_path=args.inference_report,
        repo_root=args.repo_root,
        original_metrics_path=args.original_metrics,
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.replayed.jsonl"
    reference_path = output / "references.posthoc_subset.jsonl"
    metrics_path = output / "metrics.replayed.json"
    report_path = output / "forensic_replay.json"
    _write_ledgers(prediction_path, predictions)
    _write_ledgers(reference_path, references)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["replayed_artifacts"] = {
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
        },
        "posthoc_references": {
            "path": reference_path.name,
            "sha256": file_sha256(reference_path),
        },
        "metrics": {
            "path": metrics_path.name,
            "sha256": file_sha256(metrics_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "claim_scope": report["claim_scope"],
                "counts": report["counts"],
                "replayed_headline_metrics": report["replayed_headline_metrics"],
                "output": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
