"""Categorize ESC-50 files and run MOSS zero-shot captioning on real audio."""
import pandas as pd, os, shutil, json

# Categorize ESC-50 files by type for our pool
df1 = pd.read_parquet('/tmp/real_audio/esc50/data/train-00000-of-00002-2f1ab7b824ec751f.parquet')
df2 = pd.read_parquet('/tmp/real_audio/esc50/data/train-00001-of-00002-27425e5c1846b494.parquet')
df = pd.concat([df1, df2])

kind_map = {
    'sfx': ['dog', 'cat', 'rooster', 'pig', 'cow', 'frog', 'door_wood_knock',
            'door_wood_creaks', 'can_opening', 'glass_breaking', 'chainsaw', 'siren',
            'car_horn', 'church_bells', 'fireworks', 'clapping', 'footsteps', 'laughing'],
    'ambience': ['rain', 'wind', 'sea_waves', 'thunderstorm', 'crackling_fire',
                 'water_drops', 'insects', 'crickets', 'chirping_birds'],
}
cat_to_kind = {}
for kind, cats in kind_map.items():
    for c in cats:
        cat_to_kind[c] = kind
for c in df['category'].unique():
    if c not in cat_to_kind:
        cat_to_kind[c] = 'sfx'

for kind in ['sfx', 'ambience']:
    os.makedirs(f'/tmp/real_audio/esc50_categorized/{kind}', exist_ok=True)

for _, row in df.iterrows():
    fname = row['filename']
    cat = row['category']
    kind = cat_to_kind.get(cat, 'sfx')
    src = f'/tmp/real_audio/esc50_wav/{fname}'
    dst = f'/tmp/real_audio/esc50_categorized/{kind}/{fname}'
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

for kind in ['sfx', 'ambience']:
    files = os.listdir(f'/tmp/real_audio/esc50_categorized/{kind}')
    print(f'{kind}: {len(files)} files')

cat_map = {row['filename']: row['category'] for _, row in df.iterrows()}
with open('/tmp/real_audio/esc50_category_map.json', 'w') as f:
    json.dump(cat_map, f)
print('Category map saved')
