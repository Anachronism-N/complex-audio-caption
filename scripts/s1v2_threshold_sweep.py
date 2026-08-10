"""Compatibility launcher for leakage-safe S1 boundary threshold calibration.

The historical script swept the reported validation fold and highlighted
boundary MAE at 1% recall. This launcher now delegates to the unified runner,
which selects a threshold on a source-disjoint calibration fold by event F1
and evaluates the reported validation fold only after selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sceneledger.cli.train_slots import main as train_slots_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    args, forwarded = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = args.config or str(root / "configs/model/s1_event_slots.yaml")
    return train_slots_main(
        [
            "--config",
            config,
            "--evaluate-checkpoint",
            args.checkpoint,
            "--activity-weight",
            "0",
            "--activity-cost-weight",
            "0",
            "--primary-decode-mode",
            "boundary",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
