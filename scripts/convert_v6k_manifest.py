"""Convert v6k real-mix manifest to sceneledger-compatible format."""
import json

entries = [json.loads(l) for l in open('data/derived/real_mix_v6_1k/manifest.jsonl')]
out = []
for e in entries:
    sid = e['scene_id']
    sources = e['sources']
    scene = {
        'scene_id': sid,
        'seed': 0,
        'duration': e['duration'],
        'template': e['scene_name'],
        'sources': [
            {
                'source_id': f'S{j+1}',
                'kind': s['type'],
                'path': f"real:{s['role']}",
                'onset': s['onset'],
                'gain_db': s.get('gain_db', 0),
                'repeat': 1,
                'repeat_gap_s': 0.0,
                't60_sec': None,
            }
            for j, s in enumerate(sources)
        ],
        'conditions': {'t60_sec': None, 'echo_delay_ms': None, 'echo_atten_db': None},
        'supervision': {'activity_threshold': 0.05, 'resolution_s': 0.1, 'merge_threshold_s': 0.3},
    }
    out.append({
        'scene': scene,
        'mixture_path': e['audio_path'],
        'stem_paths': {},
        'mixture_hash': '',
        'dry_mixture_hash': '',
        'stem_hashes': {},
        'activity_hashes': {},
        'target_ledger': e['ledger'],
        'sample_rate': 16000,
    })

with open('data/derived/real_mix_v6_1k/manifest_compat.jsonl', 'w') as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

print(f'Converted {len(out)} entries')
from sceneledger.data.manifests import read_manifest
entries = read_manifest('data/derived/real_mix_v6_1k/manifest_compat.jsonl')
print(f'Verified: {len(entries)} entries, first: {entries[0].scene["scene_id"]}')
