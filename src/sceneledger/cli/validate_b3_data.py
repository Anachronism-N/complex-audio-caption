"""Build the fail-closed acceptance summary for a B3-valid data release."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sceneledger.data.reproduction import validate_b3_data_release


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-validate-b3-data")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--render-report", required=True)
    parser.add_argument("--sft-metadata", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args(argv)

    summary = validate_b3_data_release(
        source_report_path=args.source_report,
        render_report_path=args.render_report,
        sft_metadata_path=args.sft_metadata,
        train_manifest_path=args.train_manifest,
        val_manifest_path=args.val_manifest,
        expected_samples=args.expected_samples,
        allow_unknown_license=args.allow_unknown_license,
        git_commit=_git_commit(),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
