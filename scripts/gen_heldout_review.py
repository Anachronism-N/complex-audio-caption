"""Deprecated guard for the invalid real_mix_v6 index-slice review."""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "The rv6_0181..0200 package is not held out: 15/20 samples were used "
        "during training and dry-source identities are unavailable. Use "
        "sceneledger-model-review on the certified Real-Complex test split instead."
    )


if __name__ == "__main__":
    raise SystemExit(main())
