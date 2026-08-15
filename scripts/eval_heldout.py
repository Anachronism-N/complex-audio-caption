"""Deprecated: the legacy index-slice evaluation was not held out.

The historical implementation selected ``manifest[180:]`` although the
trainer used a group split.  Fifteen of those twenty samples were therefore
seen during training.  Keep this filename as an explicit guard so the invalid
experiment cannot be repeated accidentally.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "This evaluation is invalid: manifest[180:] is not the trainer's held-out "
        "split. Run scripts/run_b3_real_complex_anchor.sh on a passed three-fold "
        "data contract instead; use sceneledger-audit-result for result certification."
    )


if __name__ == "__main__":
    raise SystemExit(main())
