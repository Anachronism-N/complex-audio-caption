# 下一步实验：D0-SOURCE-SMOKE-v1 真实音源池验收

## 0. 决策

本实验**不需要新增代码**。当前 `agent/s1-valid-experiments` 分支已经实现：

- 五类单源 catalog 规范化；
- 逐文件解码、字节与 decoded-PCM 哈希；
- 时长、RMS、峰值、削波、静音和非有限值检查；
- decoded-audio 精确去重；
- license、真实文本、vocal 逐字歌词检查；
- source 数、独立 source group 数和总时长配额；
- 稳定 `source_pool_id` 和 render 前 fail-closed gate。

下一步的未知量不是模型结构，而是“我们是否真的拥有一批可解码、可授权、语义正确且足够多样的五类音源”。因此先执行一次 CPU-only 的真实音源 smoke 验收。报告通过前，不渲染 mixture，不导出 SFT，不训练 B3/S1，也不继续比较 slot-aware、RL 或 agent 重写。

只有出现以下情况才重新增加代码：

1. 满足文档契约的合法数据被门禁错误拒绝；
2. 相同 catalog、音频和配置重复运行却得到不同 `source_pool_id`；
3. 报告缺少定位实际失败文件所需的信息；
4. 后续明确决定把视频切分、分离、teacher 标注纳入可复现数据生产流水线。

普通的缺文件、标签错误、授权未知、配额不足、静音、削波或重复音频都是数据问题，不修改代码绕过。

## 1. 实验目标和成功条件

实验编号：`D0-SOURCE-SMOKE-v1`

目标：冻结一份能支持后续 100-scene renderer smoke 的真实单源池。

自动验收必须同时满足：

```text
source_readiness_report.json.pass == true
source_readiness_report.json.failed_checks == []
source_readiness_report.json.profile == "smoke"
source_readiness_report.json.source_pool_id != ""
n_sources == n_audio_ok == n_unique_decoded_audio
```

自动验收通过后还必须完成人工试听：每类固定随机抽查至少 10 条，并检查全部 vocal。任意音频与标签不一致、歌词并非逐字可听或存在明显分离泄漏，均视为失败；修正数据后重新执行本实验。

本实验的成功只表示“音源池可进入 renderer smoke”，不表示训练数据已经准备好，更不表示模型实验有效。

## 2. 固定代码和环境

在 Linux 数据服务器上执行。该阶段不需要 GPU。

```bash
set -euo pipefail

REPO=/path/to/complex-audio-caption
cd "${REPO}"

git fetch origin
git switch agent/s1-valid-experiments
git pull --ff-only origin agent/s1-valid-experiments
git status --short --branch

python -m pip install -e ".[dev,audio]"
python -m pytest tests/unit/test_data_pipeline.py -q
```

`git status` 必须没有未提交修改，测试必须通过。不要在服务器上把最新 `main` 合并进该分支；当前 `main` 的 slot-aware 改动与数据门禁分支存在训练入口冲突，但不影响本次 source-only 实验。

建立不会被后续正式 release 覆盖的独立目录，并记录运行身份：

```bash
export SOURCE_ROOT=/data/b3
export WORK_DIR=/data/runs/b3_smoke

mkdir -p "${SOURCE_ROOT}/audio" "${SOURCE_ROOT}/catalogs" "${WORK_DIR}"
git rev-parse HEAD > "${WORK_DIR}/code_commit.txt"
python --version > "${WORK_DIR}/python_version.txt"
python -m pip freeze > "${WORK_DIR}/pip_freeze.txt"
cp configs/data/source_readiness.yaml "${WORK_DIR}/source_readiness.frozen.yaml"
```

不要复用旧 `data/derived/b3_unified` 的 synthetic 500 条数据。

## 3. 准备五类真实单源

推荐目录结构：

```text
/data/b3/
├── audio/
│   ├── speech/
│   ├── vocal/
│   ├── music/
│   ├── sfx/
│   └── ambience/
├── catalogs/
│   ├── speech.csv
│   ├── vocal.csv
│   ├── music.csv
│   ├── sfx.csv
│   └── ambience.csv
└── source_catalog.csv
```

优先使用 WAV 或 FLAC。每个文件应尽量只对应一个可描述的 source；不要把未经检查的复杂混音直接登记成单源。

### 3.1 Catalog 字段

CSV 表头固定为：

```csv
path,kind,text,source_group,identity,language,verbatim,license,dataset
```

字段契约：

| 字段 | 要求 |
|---|---|
| `path` | 相对于 `/data/b3/audio` 的路径，或可读绝对路径 |
| `kind` | `speech/vocal/music/sfx/ambience` 之一 |
| `text` | 音频中真实可听内容；所有类别都禁止空值和占位描述 |
| `source_group` | 原说话人、歌曲、原视频或原始录音 ID；同源切片必须相同 |
| `identity` | 可选的 speaker/vocal identity，例如 `S1`、`V1` |
| `language` | speech/vocal 建议填写 ISO 语言码 |
| `verbatim` | vocal 必须为 `true`；speech 若是逐字转录也填 `true` |
| `license` | 明确许可证或真实的 `internal-authorized`；禁止 `unknown/TBD` |
| `dataset` | 数据集名称和版本 |

示例：

```csv
path,kind,text,source_group,identity,language,verbatim,license,dataset
speech/19-198-0001.flac,speech,A MAN SAID SOMETHING,speaker-19,S1,en,true,CC BY 4.0,LibriSpeech-dev-clean
vocal/song01_001.wav,vocal,我们一起走过漫长的夜,song-01,V1,zh,true,internal-authorized,internal-vocal-v1
music/song02_inst.wav,music,soft piano accompaniment,song-02,,,false,internal-authorized,internal-music-v1
sfx/glass_001.wav,sfx,a glass breaks,recording-glass-001,,,false,CC0,internal-sfx-v1
ambience/room_001.wav,ambience,steady indoor room ambience,recording-room-001,,,false,CC0,internal-ambience-v1
```

不要照抄示例文本；每一行都要与对应音频一致。

### 3.2 Smoke 最低配额

| kind | 最少 sources | 最少 source groups | 最少总时长 |
|---|---:|---:|---:|
| speech | 20 | 10 | 30 s |
| vocal | 20 | 10 | 40 s |
| music | 20 | 10 | 120 s |
| sfx | 40 | 20 | 20 s |
| ambience | 20 | 10 | 120 s |

单文件允许时长：

| kind | 最短 | 最长 |
|---|---:|---:|
| speech | 0.3 s | 30 s |
| vocal | 1 s | 60 s |
| music | 3 s | 120 s |
| sfx | 0.05 s | 30 s |
| ambience | 3 s | 120 s |

不要用复制文件、改文件名或从同一原视频密集切片的方式凑数量。decoded waveform 完全相同会被自动拒绝；同一原媒体的多个切片必须共享 `source_group`，因此也不能虚增独立 group 数量。

### 3.3 可用数据与互联网爬取数据

speech 可以先用仓库内的 LibriSpeech 下载与登记脚本：

```bash
cd "${REPO}"
LIBRISPEECH_SUBSET=dev-clean \
LIBRISPEECH_ROOT=/data/librispeech \
bash scripts/data/download_librispeech.sh
```

该脚本生成的 `/data/librispeech/source_catalog_dev-clean.jsonl` 已包含绝对音频路径，合并五类 catalog 时可以直接作为一个 `--input`，无需复制音频或改写路径。

Opencpop 等需要申请或有非商业限制的数据，只能按官方许可获得后登记。音乐、SFX、ambience 和 vocal 可以使用内部已授权数据或许可兼容的公开数据。

B站、Instagram、TikTok 等爬取数据目前没有 ground truth，**不能直接进入本实验的监督 source pool**。只有同时完成以下操作的片段才能登记：

1. 确认研究使用和再分发/派生处理的授权状态；
2. 切出或分离为可核验的单源片段；
3. 人工核对 `kind` 和 `text`；
4. vocal 写入实际逐字歌词并设置 `verbatim=true`；
5. 用原视频 ID 作为 `source_group`；
6. 试听分离残留，不能把明显混入的 speech/music/SFX 当成纯净单源。

未完成这些条件的数据可以保留给后续无监督/teacher-student 研究，但不能用于当前基线数据复现。

## 4. 合并和预检查 catalog

如果五类数据分别维护，使用已有 CLI 合并。以下示例假设各 CSV 的 `path` 都相对于 `/data/b3/audio`：

```bash
cd "${REPO}"

python -m sceneledger.cli.prepare_sources \
  --input /data/b3/catalogs/speech.csv \
  --input /data/b3/catalogs/vocal.csv \
  --input /data/b3/catalogs/music.csv \
  --input /data/b3/catalogs/sfx.csv \
  --input /data/b3/catalogs/ambience.csv \
  --audio-root /data/b3/audio \
  --output /data/b3/source_catalog.jsonl \
  --report /data/b3/source_catalog.precheck.json \
  --require-kind speech \
  --require-kind vocal \
  --require-kind music \
  --require-kind sfx \
  --require-kind ambience
```

此命令必须成功退出。不要使用 `--allow-missing`。检查：

```bash
python - <<'PY'
import json
from pathlib import Path

p = json.loads(Path('/data/b3/source_catalog.precheck.json').read_text())
print('n_sources:', p['n_sources'])
print('kinds:', p['kinds'])
print('source_groups:', p['source_groups'])
print('licenses:', p['licenses'])
print('all_files_verified:', p['all_files_verified'])
PY
```

`all_files_verified` 必须为 `true`，五类计数都必须非零。配额和音频质量由下一阶段统一检查。

## 5. 只运行 source gate

第一次运行：

```bash
cd "${REPO}"

SOURCE_CATALOG=/data/b3/source_catalog.jsonl \
SOURCE_AUDIO_ROOT=/data/b3/audio \
SOURCE_PROFILE=smoke \
WORK_DIR=/data/runs/b3_smoke \
N_SAMPLES=100 \
STAGE=sources \
bash scripts/run_b3_data.sh 2>&1 | tee /data/runs/b3_smoke/sources.log
```

`N_SAMPLES=100` 在此阶段只负责固定后续 smoke 身份；`STAGE=sources` 不会渲染 100 条 mixture，也不需要 TAG checkpoint 或 GPU。

预期产物：

```text
/data/runs/b3_smoke/
├── code_commit.txt
├── python_version.txt
├── pip_freeze.txt
├── source_readiness.frozen.yaml
├── source_catalog.jsonl
├── source_catalog_report.json
├── source_inventory.jsonl
├── source_readiness_report.json
└── sources.log
```

## 6. 自动结果判定

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

p = json.loads(
    Path('/data/runs/b3_smoke/source_readiness_report.json').read_text()
)
print('pass:', p['pass'])
print('profile:', p['profile'])
print('source_pool_id:', p['source_pool_id'])
print('n_sources:', p['n_sources'])
print('n_audio_ok:', p['n_audio_ok'])
print('n_unique_decoded_audio:', p['n_unique_decoded_audio'])
print('failed_checks:', p['failed_checks'])
print('kinds:')
for kind, row in p['kinds'].items():
    print(' ', kind, row)
if p['failed_checks']:
    print('failure details:')
    for check in p['checks']:
        if not check['pass']:
            print(' ', check['name'], check['detail'])
PY
```

常见失败及处理：

| `failed_checks` | 原因 | 处理 |
|---|---|---|
| `all_audio_decoded_and_quality_checked` | 解码失败、空音频、近静音、时长越界、严重削波 | 修复/替换具体文件，不能降低阈值掩盖问题 |
| `decoded_audio_unique` | 不同路径包含相同 decoded PCM | 删除重复条目；不要只改文件名 |
| `all_licenses_known` | license 为空或仍是 placeholder | 查明许可；无法确认则移出监督池 |
| `all_vocal_lyrics_verbatim` | vocal 缺逐字歌词或 `verbatim!=true` | 补真实歌词；听不清/无词人声改为 music |
| `<kind>_source_quota` | 该类文件数不足 | 增加真实独立音频 |
| `<kind>_group_quota` | 独立说话人/歌曲/原媒体数不足 | 增加独立来源，不能拆分同源伪造 group |
| `<kind>_duration_quota` | 该类总时长不足 | 增加合格时长 |

修改 catalog 或 metadata 后重新运行完整 `STAGE=sources`。仅在 catalog 不变、只替换同一路径音频文件时可以运行：

```bash
cd "${REPO}"
SOURCE_PROFILE=smoke \
WORK_DIR=/data/runs/b3_smoke \
STAGE=source-audit \
bash scripts/run_b3_data.sh 2>&1 | tee /data/runs/b3_smoke/source-audit.log
```

失败时只修复音源池，不开始 renderer 或训练。

## 7. 人工试听验收

自动报告通过后，生成固定随机抽样表。该命令抽取每类最多 10 条，并自动包含全部 vocal：

```bash
python - <<'PY'
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

inventory = Path('/data/runs/b3_smoke/source_inventory.jsonl')
output = Path('/data/runs/b3_smoke/source_manual_audit.csv')
seed = 20260810

by_kind = defaultdict(list)
for line in inventory.read_text(encoding='utf-8').splitlines():
    if line.strip():
        row = json.loads(line)
        by_kind[row['kind']].append(row)

selected = {}
for kind, rows in sorted(by_kind.items()):
    rows = sorted(rows, key=lambda x: (x['source_group'], x['path']))
    random.Random(f'{seed}:{kind}').shuffle(rows)
    chosen = rows if kind == 'vocal' else rows[:10]
    for row in chosen:
        selected[row['path']] = row

fields = [
    'path', 'kind', 'text', 'source_group', 'license',
    'audio_audible', 'kind_correct', 'text_correct',
    'lyrics_verbatim', 'separation_clean', 'reviewer',
    'reviewed_at', 'notes'
]
with output.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in sorted(selected.values(), key=lambda x: (x['kind'], x['path'])):
        writer.writerow({key: row.get(key, '') for key in fields})

print(f'wrote {len(selected)} rows to {output}')
PY
```

人工填写规则：

- `audio_audible/kind_correct/text_correct/separation_clean` 填 `yes/no`；
- vocal 的 `lyrics_verbatim` 必须填 `yes`；其他类型填 `n/a`；
- `reviewer` 和 `reviewed_at` 不得为空；
- 任意一个必需项为 `no`，就从 catalog 中修正或移除该数据，然后重新执行 source gate；
- 试听者不能只看文件名判断，必须实际播放完整片段。

## 8. 冻结与回传

自动和人工检查全部通过后冻结哈希：

```bash
cd /data/runs/b3_smoke
sha256sum \
  source_catalog.jsonl \
  source_catalog_report.json \
  source_inventory.jsonl \
  source_readiness_report.json \
  source_manual_audit.csv \
  > source_pool_artifacts.sha256
```

回传以下文件供复核：

```text
code_commit.txt
source_catalog_report.json
source_readiness_report.json
source_inventory.jsonl
source_manual_audit.csv
source_pool_artifacts.sha256
sources.log
```

不要把受版权保护的音频、内部 catalog 或包含敏感绝对路径的报告直接提交到公开 GitHub。通过项目约定的私有存储回传；如需公开实验证据，应先生成不含私有路径的汇总版本。

## 9. Go/No-Go 决策

### Go

满足以下全部条件：

1. 自动报告 `pass=true`；
2. `failed_checks=[]`；
3. `source_pool_id` 非空并已记录；
4. 人工试听所有必需项均通过；
5. 回传产物哈希验证一致。

此时停止本实验并复核报告。下一步才是检查 TAG anchor，然后单独运行 `STAGE=render` 生成 100 条真实 smoke mixture。

### No-Go

任一条件失败：

- 不运行 `STAGE=render/export/audit`；
- 不启动 B3/S1 训练；
- 不根据旧 synthetic 分数修改模型；
- 只修复报告指向的音源、metadata、授权或配额问题，再以相同实验编号追加 rerun 编号，例如 `D0-SOURCE-SMOKE-v1-r2`。

## 10. 实验记录模板

将以下内容填入服务器实验日志：

```text
experiment_id: D0-SOURCE-SMOKE-v1
operator:
started_at:
finished_at:
git_commit:
server:
python_version:
source_profile: smoke
catalog_path:
catalog_sha256:
inventory_sha256:
source_pool_id:
n_sources:
n_audio_ok:
n_unique_decoded_audio:
failed_checks:
manual_audit_rows:
manual_audit_failures:
decision: GO | NO-GO
notes:
```

相关契约还可查阅：

- `docs/16_b3_data_reproduction.md`：从 source 到最终 `dataset_id` 的完整冻结协议；
- `docs/17_source_pool_readiness.md`：source readiness 门禁设计；
- `configs/data/source_readiness.yaml`：版本化 smoke/release 配额与音频阈值；
- `configs/data/source_catalog.example.csv`：最小 catalog 示例。
