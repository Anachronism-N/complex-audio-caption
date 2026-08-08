"""Smoke test: load real MOSS-Audio-4B and run inference on 2 TAC-mini clips.

Run inside the moss-audio conda env::

    conda run -n moss-audio python scripts/smoke_moss.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ensure sceneledger + third_party/MOSS-Audio are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "MOSS-Audio"))

from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.models.target_formatter import canonical_prompt

WEIGHTS = "/tmp/moss_weights"
AUDIO_BASE = Path("/tmp/tac_mini/audio")
CLIPS = ["mix_000001.wav", "mix_000002.wav", "mix_000003.wav"]


def main() -> int:
    print(f"[smoke] loading model from {WEIGHTS} ...", flush=True)
    t0 = time.time()
    adapter = MossAdapter(MossAdapterConfig(model_path=WEIGHTS, device="cuda:0", dtype="bfloat16"))
    # trigger lazy load
    adapter._load()
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s", flush=True)

    prompt = canonical_prompt(style="brief", include_lyrics=False)
    print(f"[smoke] prompt:\n{prompt}\n", flush=True)

    for clip in CLIPS:
        path = AUDIO_BASE / clip
        if not path.exists():
            print(f"[smoke] SKIP {clip} (not found)", flush=True)
            continue
        print(f"[smoke] inferring {clip} ...", flush=True)
        t1 = time.time()
        out = adapter.infer(str(path), prompt, sample_id=clip, duration=30.0)
        dt = time.time() - t1
        print(f"[smoke] {clip} ({dt:.1f}s):", flush=True)
        print(out[:500], flush=True)
        print("---", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
