# 12 小时实验执行手册：真实音源池与 100-scene 数据闭环

## 0. 本次实验的性质

本次 12 小时窗口不用于训练新的 caption 模型，而用于回答一个更基础、也更重要的问题：

> 当前真实音源是否足以生成可复现、无明显标签错误、无 raw-source 泄漏、可以进入后续基线训练的数据？

实验分为两个串行层级：

- `D0-SOURCE-SMOKE-v1`：五类真实单源的自动验收、稳定性复跑和人工试听；
- `D1-RENDER-SMOKE-v1`：仅在 D0 与 TAG anchor 都通过时，生成 100 个真实来源的复杂 mixture，导出 SFT split 并完成最终数据审计。

本次实验不报告模型 F1，不运行 B3/S1/slot-aware/RL，不使用旧 `b3_unified` 或 `b3_5k` synthetic 数据。100-scene 数据只是数据管线 smoke，不是论文训练集。

## 1. 为什么这样安排

最新 5k synthetic 实验暴露了两个不能再忽略的问题：

1. 旧数据不是 file-backed 真实音源，无法证明 speech/music/lyrics/SFX 在真实复杂声学条件下有效；
2. 5k manifest 的实际 split 为 4983/17，报告前 500 条中 497 条属于训练集，说明“无 raw-source 泄漏”仍不足以保证验证集规模合理。

因此，本次实验除了现有自动 gate，还增加两个人工协议门槛：

- 100-scene smoke 的 validation scenes 必须在 10–30 条之间；
- 最大 source-connected component 不得超过 30 scenes。

这两个阈值是本项目的 smoke 决策阈值，不是外部论文声称。若失败，结果不是调模型，而是证明下一项代码工作应改为“在 render 前划分 raw source pool，再分别采样 train/val scenes”。

## 2. 12 小时结束时必须交付什么

最低交付是 D0 的完整证据：

```text
source_catalog_report.json
source_inventory.jsonl
source_readiness_report.json
source_manual_audit.csv
source_pool_id.txt
sources.log
source-audit-rerun.log
environment/
```

如果 D0 和 TAG anchor 都通过，增加 D1 证据：

```text
data/manifest.jsonl
data/validation_report.json
data/listen_list.csv
sft/metadata.json
sft/train_manifest.jsonl
sft/val_manifest.jsonl
sft/val_references.jsonl
data_reproduction_summary.json
split_diagnostics.json
mixture_manual_audit.csv
render.log
export.log
audit.log
artifact_sha256.txt
```

如果某个 gate 未通过，仍需提交失败报告和日志。失败但可定位的数据实验，比缺少 provenance 的高分模型实验更有价值。

## 3. 进入 12 小时窗口前的最低条件

开始计时前应尽量准备：

- 一台 Linux 服务器；D0 和 D1 都不需要 GPU；
- Python 3.10 或更高版本；
- 仓库读写权限和足够磁盘空间；
- 五类音频文件：`speech/vocal/music/sfx/ambience`；
- 每条音频的真实语义标签、`source_group`、license；
- vocal 的逐字歌词和 `verbatim=true`；
- 若希望执行 D1，已有 `TAG_SUMMARY` 且其 `pass=true`。

若开始时还没有五类 catalog，采用“数据准备路径”：12 小时目标只设为 D0，不承诺 D1。不要因为机器空闲而改跑 synthetic 训练。

## 4. 总体时间表

| 时间 | 阶段 | 必须完成 | 决策 |
|---|---|---|---|
| H0:00–H0:30 | P0 环境冻结 | 分支、依赖、测试、路径、磁盘、TAG 状态 | 环境失败则先修环境 |
| H0:30–H2:30 | P1 Catalog 定稿 | 五类合并、文件存在、metadata 与配额预检查 | 缺类/缺授权则继续整理数据 |
| H2:30–H3:30 | P2 Source gate | 逐文件解码、质量、去重、配额、`source_pool_id` | 自动失败则进入修复路径 |
| H3:30–H5:00 | P3 Source 人工试听 | 每类至少 10 条，全部 vocal | 任一关键错误即 No-Go |
| H5:00 | Gate A | D0 自动+人工+稳定性全部通过 | 决定是否允许 D1 |
| H5:00–H7:00 | P4 修复或 Render | No-Go：修数据；Go+TAG：渲染 100 scenes | 禁止绕过 TAG/source gate |
| H7:00–H8:00 | P5 Export + Audit | 冻结 train/val 和 `dataset_id` | 自动 gate 必须全通过 |
| H8:00–H9:00 | P6 Split 诊断 | validation 数量、component 和 leakage | 比例异常则禁止训练 |
| H9:00–H10:30 | P7 Mixture 试听 | 分层检查复杂场景、歌词、时间边界 | 关键错误即 No-Go |
| H10:30–H11:15 | P8 身份复核 | 哈希、ID、代码和配置冻结 | 任意不一致需解释 |
| H11:15–H12:00 | P9 打包回传 | 日志、报告、人工表和结论 | 到点停止，不追加模型实验 |

时间表是上限而非等待要求。某一步提前结束就立即进入下一步；某个 fail-closed gate 失败后，后续禁止阶段必须取消。

## 5. P0：环境冻结（H0:00–H0:30）

### 5.1 获取正确分支

必须使用包含真实数据门禁的分支，不使用当前 `main`：

```bash
set -euo pipefail

export REPO=/path/to/complex-audio-caption
cd "${REPO}"

git fetch origin
git switch agent/s1-valid-experiments
git pull --ff-only origin agent/s1-valid-experiments
git status --short --branch
```

`git status` 必须干净。记录完整 commit，不在实验中途 pull 或 merge：

```bash
export RUN_ID="b3_data_12h_$(date -u +%Y%m%dT%H%M%SZ)"
export WORK_DIR="/data/runs/${RUN_ID}"
export SOURCE_AUDIO_ROOT=/data/b3/audio
export SOURCE_CATALOG=/data/b3/source_catalog.jsonl
export SOURCE_PROFILE=smoke
export N_SAMPLES=100
export VAL_FRACTION=0.1
export SPLIT_SEED=20260808
export TAG_SUMMARY="${REPO}/runs/tag2021/reproduction_summary.json"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${WORK_DIR}/environment"
git rev-parse HEAD > "${WORK_DIR}/environment/git_commit.txt"
git status --porcelain=v1 > "${WORK_DIR}/environment/git_status.txt"
python --version > "${WORK_DIR}/environment/python_version.txt" 2>&1
df -h "${WORK_DIR}" > "${WORK_DIR}/environment/disk_before.txt"
```

后续命令默认在同一个 shell 中执行。若 SSH 断开或更换终端，必须重新设置上述环境变量，并确认 `WORK_DIR` 指向同一实验目录；不要重新生成 `RUN_ID`。

### 5.2 安装和最小测试

```bash
cd "${REPO}"
python -m pip install -e ".[dev,audio]"
python -m pip freeze > "${WORK_DIR}/environment/pip_freeze.txt"

python - <<'PY'
import numpy, scipy, soundfile, yaml
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('soundfile', soundfile.__version__)
print('pyyaml', yaml.__version__)
PY

python -m pytest tests/unit/test_data_pipeline.py -q \
  2>&1 | tee "${WORK_DIR}/environment/data_pipeline_tests.log"
```

测试失败则先停止数据生成。不要用 `--allow-missing`、`--allow-placeholder-lyrics` 或 `ALLOW_UNKNOWN_LICENSE=1` 让测试/数据绕过门禁。

### 5.3 记录 TAG 状态但不阻塞 D0

```bash
set +e
python scripts/repro/require_anchor_pass.py "${TAG_SUMMARY}" \
  2>&1 | tee "${WORK_DIR}/environment/tag_gate.log"
TAG_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${TAG_RC}" > "${WORK_DIR}/environment/tag_gate.exit_code"
```

- `TAG_RC=0`：若 D0 通过，可执行 D1；
- 非 0：仍可执行 D0，但禁止 `render/export/audit`；最终记录 `D1=BLOCKED_BY_TAG`。

## 6. P1：真实 catalog 定稿（H0:30–H2:30）

### 6.1 输入契约

CSV/JSONL 每条记录必须包含：

| 字段 | 规则 |
|---|---|
| `path` | 可解码单源音频；相对路径以 `SOURCE_AUDIO_ROOT` 为基准 |
| `kind` | `speech/vocal/music/sfx/ambience` 之一 |
| `text` | 与真实可听内容一致，禁止 placeholder |
| `source_group` | 原说话人、歌曲、原视频或原录音 ID；同源派生片段必须一致 |
| `identity` | 可选 speaker/singer identity |
| `language` | speech/vocal 建议填写 |
| `verbatim` | vocal 必须为 `true` |
| `license` | 明确许可证或确有授权时填写 `internal-authorized` |
| `dataset` | 数据集名称和版本 |

固定表头：

```csv
path,kind,text,source_group,identity,language,verbatim,license,dataset
```

互联网爬取但无 ground truth 的视频不能直接进入本实验。必须先确认许可、切分/分离、人工核对标签；vocal 必须逐字核对歌词，并以原视频 ID 作为 `source_group`。无法确认授权或语义的文件移出监督池，不要写成 `internal-authorized` 或猜测标签。

### 6.2 Smoke 配额

| kind | 最少 sources | 最少 source groups | 最少总时长 | 单文件时长范围 |
|---|---:|---:|---:|---:|
| speech | 20 | 10 | 30 s | 0.3–30 s |
| vocal | 20 | 10 | 40 s | 1–60 s |
| music | 20 | 10 | 120 s | 3–120 s |
| sfx | 40 | 20 | 20 s | 0.05–30 s |
| ambience | 20 | 10 | 120 s | 3–120 s |

额外自动阈值：

- RMS 不低于 `-70 dBFS`；
- clipped fraction 不高于 `0.10`；
- 所有 sample 为有限值；
- decoded PCM 不得重复；
- 每个文件必须可被 `soundfile` 解码。

### 6.3 可选 speech 下载

如果缺 speech，可使用已有 LibriSpeech 脚本：

```bash
cd "${REPO}"
set -euo pipefail
LIBRISPEECH_SUBSET=dev-clean \
LIBRISPEECH_ROOT=/data/librispeech \
bash scripts/data/download_librispeech.sh \
  2>&1 | tee "${WORK_DIR}/librispeech.log"
```

输出 `/data/librispeech/source_catalog_dev-clean.jsonl` 可以直接作为后续一个 `--input`。

### 6.4 合并五类 catalog

以下路径按服务器实际情况替换：

```bash
cd "${REPO}"
set -euo pipefail

python -m sceneledger.cli.prepare_sources \
  --input /data/librispeech/source_catalog_dev-clean.jsonl \
  --input /data/b3/catalogs/vocal.csv \
  --input /data/b3/catalogs/music.csv \
  --input /data/b3/catalogs/sfx.csv \
  --input /data/b3/catalogs/ambience.csv \
  --audio-root "${SOURCE_AUDIO_ROOT}" \
  --output "${SOURCE_CATALOG}" \
  --report "${WORK_DIR}/catalog_precheck.json" \
  --require-kind speech \
  --require-kind vocal \
  --require-kind music \
  --require-kind sfx \
  --require-kind ambience \
  2>&1 | tee "${WORK_DIR}/catalog_precheck.log"
```

不要使用 `--allow-missing`。若不使用 LibriSpeech，把第一个 `--input` 换成实际 speech catalog。

快速检查：

```bash
python - <<PY
import json
from pathlib import Path
p = json.loads(Path('${WORK_DIR}/catalog_precheck.json').read_text())
print('n_sources:', p['n_sources'])
print('kinds:', p['kinds'])
print('source_groups:', p['source_groups'])
print('licenses:', p['licenses'])
print('all_files_verified:', p['all_files_verified'])
assert p['all_files_verified'] is True
assert all(p['kinds'].get(k, 0) > 0 for k in ['speech','vocal','music','sfx','ambience'])
PY
```

## 7. P2：运行 source gate（H2:30–H3:30）

只运行 `sources`，不要运行 `all`：

```bash
cd "${REPO}"
set -o pipefail
set +e

time SOURCE_CATALOG="${SOURCE_CATALOG}" \
SOURCE_AUDIO_ROOT="${SOURCE_AUDIO_ROOT}" \
SOURCE_PROFILE="${SOURCE_PROFILE}" \
WORK_DIR="${WORK_DIR}" \
N_SAMPLES="${N_SAMPLES}" \
STAGE=sources \
bash scripts/run_b3_data.sh \
  2>&1 | tee "${WORK_DIR}/sources.log"
SOURCE_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${SOURCE_RC}" > "${WORK_DIR}/sources.exit_code"

if [[ ! -f "${WORK_DIR}/source_readiness_report.json" ]]; then
  echo "source readiness report was not produced; inspect sources.log" >&2
  exit 1
fi
```

检查结果并让失败返回非零：

```bash
python - <<PY
import json
from pathlib import Path
p = json.loads(Path('${WORK_DIR}/source_readiness_report.json').read_text())
keys = [
    'pass', 'profile', 'source_pool_id', 'n_sources',
    'n_audio_ok', 'n_unique_decoded_audio', 'failed_checks'
]
for key in keys:
    print(f'{key}:', p.get(key))
for kind, row in p['kinds'].items():
    print(kind, row)
for check in p['checks']:
    if not check['pass']:
        print('FAILED', check['name'], check['detail'])
assert p['pass'] is True
assert p['failed_checks'] == []
assert p['profile'] == 'smoke'
assert p['source_pool_id']
assert p['n_sources'] == p['n_audio_ok'] == p['n_unique_decoded_audio']
Path('${WORK_DIR}/source_pool_id.txt').write_text(p['source_pool_id'] + '\n')
PY
```

若 assert 失败，直接进入第 14 节的故障树，不执行 D1。

### 7.1 稳定性复跑

在不修改 catalog 和音频的前提下重复 audit，要求 ID 完全相同：

```bash
set -euo pipefail

cp "${WORK_DIR}/source_readiness_report.json" \
   "${WORK_DIR}/source_readiness_report.first.json"

SOURCE_PROFILE="${SOURCE_PROFILE}" \
WORK_DIR="${WORK_DIR}" \
STAGE=source-audit \
bash scripts/run_b3_data.sh \
  2>&1 | tee "${WORK_DIR}/source-audit-rerun.log"

python - <<PY
import json
from pathlib import Path
a = json.loads(Path('${WORK_DIR}/source_readiness_report.first.json').read_text())
b = json.loads(Path('${WORK_DIR}/source_readiness_report.json').read_text())
print('first:', a['source_pool_id'])
print('rerun:', b['source_pool_id'])
assert a['source_pool_id'] == b['source_pool_id']
assert a['source_catalog_sha256'] == b['source_catalog_sha256']
assert a['inventory_sha256'] == b['inventory_sha256']
PY
```

`generated_at_utc` 可以变化，三个 identity/hash 不得变化。

## 8. P3：source 人工试听（H3:30–H5:00）

自动 gate 无法判断标签是否真的符合声音。生成固定抽样表：每类至少 10 条，并包含全部 vocal。

```bash
python - <<PY
import csv, json, random
from collections import defaultdict
from pathlib import Path

inventory = Path('${WORK_DIR}/source_inventory.jsonl')
output = Path('${WORK_DIR}/source_manual_audit.csv')
by_kind = defaultdict(list)
for line in inventory.read_text(encoding='utf-8').splitlines():
    if line.strip():
        row = json.loads(line)
        by_kind[row['kind']].append(row)

selected = {}
for kind, rows in sorted(by_kind.items()):
    rows = sorted(rows, key=lambda x: (x['source_group'], x['path']))
    random.Random(f'20260811:{kind}').shuffle(rows)
    chosen = rows if kind == 'vocal' else rows[:10]
    for row in chosen:
        selected[row['path']] = row

fields = [
    'path','kind','text','source_group','license',
    'audio_audible','kind_correct','text_correct','lyrics_verbatim',
    'separation_clean','reviewer','reviewed_at','notes'
]
with output.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in sorted(selected.values(), key=lambda x: (x['kind'], x['path'])):
        writer.writerow({k: row.get(k, '') for k in fields})
print('audit rows:', len(selected), '->', output)
PY
```

逐条播放完整音频并填写：

- `audio_audible`：非静音、没有严重失真；
- `kind_correct`：类别正确；
- `text_correct`：描述/转录与声音一致；
- `lyrics_verbatim`：vocal 必须逐字可听，其他类型填 `n/a`；
- `separation_clean`：若来自分离模型，不存在足以改变标签的串音；
- `reviewer/reviewed_at`：必填。

自动核对人工表：

```bash
python - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path('${WORK_DIR}/source_manual_audit.csv').open(encoding='utf-8-sig')))
assert rows, 'empty manual audit'
failures = []
for row in rows:
    required = ['audio_audible','kind_correct','text_correct','separation_clean']
    failures += [(row['path'], key, row[key]) for key in required if row[key].lower() != 'yes']
    if row['kind'] == 'vocal' and row['lyrics_verbatim'].lower() != 'yes':
        failures.append((row['path'], 'lyrics_verbatim', row['lyrics_verbatim']))
    if not row['reviewer'].strip() or not row['reviewed_at'].strip():
        failures.append((row['path'], 'review_identity', 'missing'))
print('rows:', len(rows), 'failures:', len(failures))
for failure in failures[:50]: print('FAILED', failure)
assert not failures
PY
```

注意：上面的检查只有在人工实际填写后才有意义。不得批量填 `yes` 代替试听。

## 9. Gate A：是否允许 D1（H5:00）

D0 只有同时满足以下条件才为 `GO`：

1. `source_readiness_report.pass=true`；
2. `failed_checks=[]`；
3. `source_pool_id` 非空且稳定性复跑一致；
4. 自动配额全部达到 smoke 要求；
5. 人工试听没有关键失败；
6. license 和授权状态可追溯。

决策矩阵：

| D0 | TAG | 后续动作 |
|---|---|---|
| FAIL | 任意 | 修 catalog/音频，再跑 D0；禁止 D1 |
| PASS | FAIL/MISSING | 冻结 `source_pool_id`，记录 `D1=BLOCKED_BY_TAG` |
| PASS | PASS | 允许执行 D1 的 render → export → audit |

不要手工编辑 readiness report，把 `pass` 改成 true。

## 10. P4：条件式渲染 100 scenes（H5:00–H7:00）

先再次验证两个依赖：

```bash
python scripts/data/require_source_readiness_pass.py \
  "${WORK_DIR}/source_readiness_report.json" --profile smoke
python scripts/repro/require_anchor_pass.py "${TAG_SUMMARY}"
```

两条都成功后运行：

```bash
cd "${REPO}"
set -o pipefail
set +e

time SOURCE_PROFILE=smoke \
WORK_DIR="${WORK_DIR}" \
N_SAMPLES=100 \
TAG_SUMMARY="${TAG_SUMMARY}" \
STAGE=render \
bash scripts/run_b3_data.sh \
  2>&1 | tee "${WORK_DIR}/render.log"
RENDER_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${RENDER_RC}" > "${WORK_DIR}/render.exit_code"

if [[ ! -f "${WORK_DIR}/data/validation_report.json" ]]; then
  echo "render validation report was not produced; inspect render.log" >&2
  exit 1
fi
```

自动验证报告：

```bash
python - <<PY
import json
from pathlib import Path
p = json.loads(Path('${WORK_DIR}/data/validation_report.json').read_text())
for key in [
    'pass','n_entries','n_replay_ok','n_replay_fail',
    'n_stems_sum_ok','n_stems_sum_fail','n_ledger_valid',
    'n_ledger_invalid','n_audio_files_fail',
    'n_saved_reconstruction_ok','n_saved_reconstruction_fail'
]: print(key, p.get(key))
assert p['pass'] is True
assert p['n_entries'] == 100
assert p['n_replay_ok'] == 100
assert p['n_stems_sum_ok'] == 100
assert p['n_ledger_valid'] == 100
assert p['n_saved_reconstruction_ok'] == 100
assert not p['failures']
PY
```

任何 replay、stem sum、Ledger 或落盘 reconstruction 失败都停止 D1。

## 11. P5：Export 与最终自动 audit（H7:00–H8:00）

分阶段执行，不运行 `STAGE=all`：

```bash
cd "${REPO}"
set -euo pipefail

SOURCE_PROFILE=smoke \
WORK_DIR="${WORK_DIR}" \
N_SAMPLES=100 \
VAL_FRACTION=0.1 \
SPLIT_SEED=20260808 \
TAG_SUMMARY="${TAG_SUMMARY}" \
STAGE=export \
bash scripts/run_b3_data.sh \
  2>&1 | tee "${WORK_DIR}/export.log"

SOURCE_PROFILE=smoke \
WORK_DIR="${WORK_DIR}" \
N_SAMPLES=100 \
TAG_SUMMARY="${TAG_SUMMARY}" \
STAGE=audit \
bash scripts/run_b3_data.sh \
  2>&1 | tee "${WORK_DIR}/audit.log"
```

检查最终 summary：

```bash
python - <<PY
import json
from pathlib import Path
p = json.loads(Path('${WORK_DIR}/data_reproduction_summary.json').read_text())
print('pass:', p['pass'])
print('dataset_id:', p['dataset_id'])
print('source_pool_id:', p['source_pool_id'])
print('n_train/n_val:', p['n_train'], p['n_val'])
print('failed_checks:', p['failed_checks'])
for check in p['checks']:
    if not check['pass']:
        print('FAILED', check['name'], check['detail'])
assert p['pass'] is True
assert p['failed_checks'] == []
assert p['dataset_id']
assert p['source_pool_id']
PY
```

这里的自动 `pass=true` 仍不足以允许训练；还必须执行下一节的 split 规模诊断。

## 12. P6：Split 与 connected-component 诊断（H8:00–H9:00）

当前 splitter 保证共享 raw source 不跨 train/val，但在 source 重用过多时可能形成巨大连通分量，导致验证集极小。运行额外诊断：

```bash
python - <<PY
import json
from pathlib import Path
from sceneledger.data.datamodule import _source_components, source_leakage
from sceneledger.data.manifests import read_manifest

root = Path('${WORK_DIR}')
entries = read_manifest(root / 'data/manifest.jsonl')
train = read_manifest(root / 'sft/train_manifest.jsonl')
val = read_manifest(root / 'sft/val_manifest.jsonl')
components = _source_components(entries)
sizes = sorted((len(x) for x in components), reverse=True)
leaked = sorted(source_leakage(train, val))
payload = {
    'n_total': len(entries),
    'n_train': len(train),
    'n_val': len(val),
    'val_fraction_actual': len(val) / max(1, len(entries)),
    'n_source_components': len(components),
    'largest_component_scenes': sizes[0] if sizes else 0,
    'largest_component_fraction': sizes[0] / max(1, len(entries)) if sizes else 0,
    'component_sizes_desc': sizes,
    'source_leakage_count': len(leaked),
    'source_leakage_preview': leaked[:20],
}
(root / 'split_diagnostics.json').write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
assert payload['source_leakage_count'] == 0
assert 10 <= payload['n_val'] <= 30, 'validation size is unusable for 100-scene smoke'
assert payload['largest_component_scenes'] <= 30, 'giant source-connected component'
PY
```

若自动 data summary 通过但这里失败，结论写为：

```text
D1_AUTOMATION=PASS
D1_SPLIT_PROTOCOL=NO-GO
next_code_change=pre-split raw source groups before rendering train/val scenes
```

不要通过随机换 seed 挑出一个好看的 split。先保存失败的 component 证据，再设计结构性修复。

## 13. P7：Mixture 分层试听（H9:00–H10:30）

从 manifest 固定选取：每种 template、最高 overlap、最高 T60、最低 SNR、最多 sources 的样本。

```bash
python - <<PY
import csv, json, random
from collections import defaultdict
from pathlib import Path

root = Path('${WORK_DIR}')
rows = [json.loads(x) for x in (root/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
selected = {}

by_template = defaultdict(list)
for row in rows:
    by_template[row['scene']['template']].append(row)
for template, group in sorted(by_template.items()):
    group = sorted(group, key=lambda x: x['scene']['scene_id'])
    random.Random(f'20260811:{template}').shuffle(group)
    for row in group[:2]: selected[row['scene']['scene_id']] = row

def add_extreme(key, values, n=5, reverse=True):
    valid = [row for row in rows if values(row) is not None]
    valid.sort(key=lambda row: (values(row), row['scene']['scene_id']), reverse=reverse)
    for row in valid[:n]: selected[row['scene']['scene_id']] = row

add_extreme('overlap', lambda r: r['target_ledger'].get('conditions',{}).get('overlap_ratio'))
add_extreme('t60', lambda r: r['target_ledger'].get('conditions',{}).get('t60_sec'))
add_extreme('snr', lambda r: r['target_ledger'].get('conditions',{}).get('snr_db'), reverse=False)
add_extreme('sources', lambda r: len(r['scene'].get('sources',[])))

fields = [
    'scene_id','template','mixture_path','duration','n_sources','n_events',
    'overlap_ratio','t60_sec','snr_db','expected_events',
    'all_events_audible','captions_correct','lyrics_verbatim',
    'timestamps_reasonable','artifacts_acceptable','reviewer','reviewed_at','notes'
]
with (root/'mixture_manual_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for sid,row in sorted(selected.items()):
        cond=row['target_ledger'].get('conditions',{})
        writer.writerow({
            'scene_id':sid,
            'template':row['scene']['template'],
            'mixture_path':str((root/'data'/row['mixture_path']).resolve()),
            'duration':row['scene']['duration'],
            'n_sources':len(row['scene'].get('sources',[])),
            'n_events':len(row['target_ledger'].get('events',[])),
            'overlap_ratio':cond.get('overlap_ratio'),
            't60_sec':cond.get('t60_sec'),
            'snr_db':cond.get('snr_db'),
            'expected_events':json.dumps(row['target_ledger'].get('events',[]),ensure_ascii=False),
        })
print('selected scenes:',len(selected))
PY
```

实际播放 `mixture_path`，填写五个判断字段：

- 所有 target event 是否真实可听，特别是低 SNR/强重叠事件；
- speech/music/SFX 描述是否正确；
- `<lys>` 是否与真实演唱逐字一致；
- 0.1 s 边界是否基本覆盖可听区间，有无明显超过 0.2 s 的系统性偏移；
- RIR/echo/增益/重复事件是否产生不可接受的失真或标签矛盾。

关键错误零容忍：不存在的事件、错误歌词、错误类别、不可听却仍有监督、train/val raw-source 泄漏。若只发现主观边界尾音争议，记录具体 scene/event 和建议 uncertainty，不静默修改指标。

人工填写后执行完整性检查：

```bash
python - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path('${WORK_DIR}/mixture_manual_audit.csv').open(encoding='utf-8-sig')))
assert rows, 'empty mixture audit'
required = [
    'all_events_audible','captions_correct','lyrics_verbatim',
    'timestamps_reasonable','artifacts_acceptable'
]
failures = []
for row in rows:
    for key in required:
        value = row[key].strip().lower()
        if key == 'lyrics_verbatim' and value == 'n/a':
            continue
        if value != 'yes':
            failures.append((row['scene_id'], key, row[key]))
    if not row['reviewer'].strip() or not row['reviewed_at'].strip():
        failures.append((row['scene_id'], 'review_identity', 'missing'))
print('rows:', len(rows), 'failures:', len(failures))
for failure in failures[:50]: print('FAILED', failure)
assert not failures
PY
```

没有可听歌词事件的 mixture 可把 `lyrics_verbatim` 填为 `n/a`；只要包含 vocal/`<lys>`，该项必须填 `yes`。

## 14. 故障树与剩余时间使用

### 14.1 Source gate 失败

| 失败项 | 处理 | 重跑 |
|---|---|---|
| decode/quality | 替换或重新导出具体音频 | catalog 不变可 `STAGE=source-audit` |
| duplicate PCM | 删除重复记录，不允许仅改名 | `STAGE=sources` |
| unknown license | 查明许可或移出监督池 | `STAGE=sources` |
| vocal verbatim | 补真实歌词；听不清/无词人声改为 music | `STAGE=sources` |
| source/group/duration quota | 增加真实独立来源 | `STAGE=sources` |
| manual semantic failure | 修正标签或移除音频 | `STAGE=sources` |

剩余时间全部用于修复并形成 `r2/r3` 报告，不转去训练。

### 14.2 TAG 未通过

D0 可以完成，D1 必须停止。不要伪造 `reproduction_summary.json` 或绕过 `require_anchor_pass.py`。若 TAG 数据和环境已经准备好，可按 `docs/13_anchor_first_tag_reproduction.md` 使用剩余时间继续 anchor；否则打包 D0 即可。

### 14.3 Render validation 失败

保存 `validation_report.json`、manifest、具体失败 scene 和日志。不要执行 export。根据失败类型判断是 file source、renderer、activity mask、stem sum 还是落盘 PCM 问题。

### 14.4 Split 诊断失败

保存 `split_diagnostics.json`。这会直接确定下一项工程任务：实现 source-group 预划分和分别渲染，而不是调 seed、调模型或继续扩大 synthetic 数据。

### 14.5 12 小时到点仍未通过

立即停止新增阶段。保留当前所有报告和日志，填写 No-Go 原因。不要为了“跑完”而取消检查、缩短人工试听或改用 synthetic 数据。

## 15. P8/P9：冻结、打包与回传（H10:30–H12:00）

### 15.1 写实验结论

在 `${WORK_DIR}/experiment_result.md` 中填写：

```text
experiment_id: D0-SOURCE-SMOKE-v1 / D1-RENDER-SMOKE-v1
operator:
start_utc:
end_utc:
git_commit:
source_pool_id:
dataset_id:
D0_automatic: PASS | FAIL
D0_manual: PASS | FAIL
D0_identity_rerun: PASS | FAIL
TAG_anchor: PASS | FAIL | MISSING
D1_render: PASS | FAIL | BLOCKED
D1_export_audit: PASS | FAIL | BLOCKED
D1_split_protocol: PASS | FAIL | BLOCKED
D1_manual: PASS | FAIL | BLOCKED
n_sources:
n_audio_ok:
n_train:
n_val:
largest_component_scenes:
source_leakage_count:
critical_manual_failures:
final_decision: GO_TO_RELEASE_DATA | REPAIR_SOURCE_POOL | FIX_SPLITTER | FIX_RENDERER | BLOCKED_BY_TAG
notes:
```

### 15.2 生成哈希

如果只完成 D0：

```bash
cd "${WORK_DIR}"
sha256sum \
  source_catalog.jsonl \
  source_catalog_report.json \
  source_inventory.jsonl \
  source_readiness_report.json \
  source_manual_audit.csv \
  source_pool_id.txt \
  > artifact_sha256.txt
```

如果完成 D1，追加：

```bash
cd "${WORK_DIR}"
sha256sum \
  data/manifest.jsonl \
  data/validation_report.json \
  sft/metadata.json \
  sft/train_manifest.jsonl \
  sft/val_manifest.jsonl \
  sft/val_references.jsonl \
  data_reproduction_summary.json \
  split_diagnostics.json \
  mixture_manual_audit.csv \
  >> artifact_sha256.txt
```

### 15.3 元数据打包

不要把原始音频或受版权保护的 mixture 提交到公开 GitHub。仅打包日志与 metadata，通过私有存储回传：

```bash
tar -C "${WORK_DIR}" -czf "${WORK_DIR}_metadata.tar.gz" \
  environment \
  source_catalog_report.json \
  source_inventory.jsonl \
  source_readiness_report.json \
  source_manual_audit.csv \
  source_pool_id.txt \
  sources.log \
  source-audit-rerun.log \
  artifact_sha256.txt \
  experiment_result.md \
  $(test -f "${WORK_DIR}/data/validation_report.json" && echo data/validation_report.json) \
  $(test -f "${WORK_DIR}/sft/metadata.json" && echo sft/metadata.json) \
  $(test -f "${WORK_DIR}/data_reproduction_summary.json" && echo data_reproduction_summary.json) \
  $(test -f "${WORK_DIR}/split_diagnostics.json" && echo split_diagnostics.json) \
  $(test -f "${WORK_DIR}/mixture_manual_audit.csv" && echo mixture_manual_audit.csv)
```

报告内可能含绝对路径和内部 dataset/license 信息，公开前需要脱敏；但脱敏版本不能取代私有原始报告的复现审计。

## 16. 最终 Go/No-Go

| 最终状态 | 含义 | 下一步 |
|---|---|---|
| D0 FAIL | 真实音源池尚不可用 | 修 source/metadata/license |
| D0 PASS，TAG BLOCKED | 音源池已冻结，但项目 anchor 未完成 | 继续 TAG 复现 |
| D0 PASS，D1 render FAIL | renderer 对真实数据仍有错误 | 修 renderer，复跑相同 source pool |
| D1 audit PASS，split FAIL | 数据生成正确但划分不可用于实验 | 实现 render 前 source-group split |
| D1 split PASS，manual FAIL | 自动结构正确但真实听感/标签错误 | 修数据或采样策略 |
| D0/D1 全 PASS | 100-scene 数据闭环成立 | 扩充到 release source profile，再生成正式数据 |

即使 D0/D1 全部通过，也不在这 100 条 smoke 数据上报告模型结论。下一阶段是满足 `release` 配额、以新目录生成正式数据并冻结正式 `dataset_id`，之后才复现 B3 baseline。

## 17. 明确禁止事项

- 不运行旧 synthetic B3/B3-5k 训练；
- 不把 100-scene smoke 当论文训练集；
- 不在全 manifest 上评测训练模型，只允许未来对冻结的 `val_manifest.jsonl` 推理；
- 不使用 `--allow-missing`、`--allow-placeholder-lyrics`、`ALLOW_UNKNOWN_LICENSE=1`；
- 不手改 JSON 报告中的 `pass`、ID、hash 或计数；
- 不通过反复换 split seed 选择最好结果；
- 不覆盖以前已经冻结的正式数据目录；
- 不把内部/版权音频提交到公开仓库。

## 18. 相关文档

- `docs/13_anchor_first_tag_reproduction.md`：TAG 2021 anchor；
- `docs/16_b3_data_reproduction.md`：完整 B3-valid 数据冻结协议；
- `docs/17_source_pool_readiness.md`：source readiness gate；
- `docs/18_next_experiment_source_smoke.md`：D0 的精简执行手册；
- `configs/data/source_readiness.yaml`：smoke/release 配额；
- `configs/data/b3_real.yaml`：真实来源 renderer 配置。
