"""Convert v6k real-mix manifest to a diagnostic-only compatibility format.

The legacy source records contain roles but not raw recording identities or
persisted stems.  The resulting manifest intentionally fails the default
training preflight and must not be used for a paper-valid training run.
"""
import json
from pathlib import Path

from sceneledger.data.manifests import read_manifest

input_path = Path("data/derived/real_mix_v6_1k/manifest.jsonl")
output_path = Path("data/derived/real_mix_v6_1k/manifest_compat.jsonl")
entries = [
    json.loads(line)
    for line in input_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
out = []
for e in entries:
    sid = e["scene_id"]
    sources = e["sources"]
    scene = {
        "scene_id": sid,
        "seed": 0,
        "duration": e["duration"],
        "template": e["scene_name"],
        "sources": [
            {
                "source_id": f"S{j + 1}",
                "kind": s["type"],
                "path": f"real:{s['role']}",
                "onset": s["onset"],
                "gain_db": s.get("gain_db", 0),
                "repeat": 1,
                "repeat_gap_s": 0.0,
                "t60_sec": None,
            }
            for j, s in enumerate(sources)
        ],
        "conditions": {"t60_sec": None, "echo_delay_ms": None, "echo_atten_db": None},
        "supervision": {
            "activity_threshold": 0.05,
            "resolution_s": 0.1,
            "merge_threshold_s": 0.3,
        },
    }
    out.append(
        {
            "scene": scene,
            "mixture_path": e["audio_path"],
            "stem_paths": {},
            "mixture_hash": "",
            "dry_mixture_hash": "",
            "stem_hashes": {},
            "activity_hashes": {},
            "target_ledger": e["ledger"],
            "sample_rate": 16000,
        }
    )

with output_path.open("w", encoding="utf-8") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Converted {len(out)} entries")
entries = read_manifest(output_path)
print(f'Verified: {len(entries)} entries, first: {entries[0].scene["scene_id"]}')
