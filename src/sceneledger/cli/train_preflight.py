"""CPU-only training authorization preflight."""

from __future__ import annotations

import argparse
import json

from sceneledger.data.training_preflight import (
    audit_training_config,
    write_training_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-train-preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-exploratory-uncontracted",
        action="store_true",
        help="authorize an uncontracted run while permanently marking it non-publication",
    )
    args = parser.parse_args(argv)
    report = audit_training_config(
        args.config,
        repo_root=args.repo_root,
        allow_exploratory_uncontracted=args.allow_exploratory_uncontracted,
    )
    write_training_preflight(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["authorized_to_train"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
