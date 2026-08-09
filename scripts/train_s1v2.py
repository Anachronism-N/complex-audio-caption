"""Compatibility launcher for the boundary-only S1a-v2 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

from sceneledger.cli.train_slots import main as train_slots_main


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return train_slots_main(
        [
            "--config",
            str(root / "configs/model/s1_event_slots.yaml"),
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
