"""Download the pinned official MUSDB18 per-track license table.

Audio access requires the user to accept the dataset terms separately; this
script intentionally downloads metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

WEBSITE_COMMIT = "b98277bfafa9fe1bb850e88b69232b803918c0ee"
TRACKLIST_SHA256 = "5e0a9252774412be23971ebfb7d384c0b2f29842eed2fa16aab69cf825b332b1"
TRACKLIST_URL = (
    "https://raw.githubusercontent.com/sigsep/website/"
    f"{WEBSITE_COMMIT}/content/datasets/assets/tracklist.csv"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MUSDB18 tracklist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(TRACKLIST_URL) as response, partial.open("wb") as handle:
        handle.write(response.read())
    observed = hashlib.sha256(partial.read_bytes()).hexdigest()
    if observed != TRACKLIST_SHA256:
        raise ValueError(
            f"MUSDB18 tracklist checksum mismatch: expected={TRACKLIST_SHA256} observed={observed}"
        )
    partial.replace(output)
    print(f"tracklist={output}")
    print("audio_download=manual; accept MUSDB18 educational-use terms first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
