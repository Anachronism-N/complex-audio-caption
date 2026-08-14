# 下一步实验：扩充音源库，而不是继续微调 mixer

更新日期：2026-08-14

## 1. 结论与当前代码审计

远端 `main` 的最新提交是 `bf9b214`，新增了 `real_mix_v4` 及其人工反馈。本地已将该提交合并。v4 虽然加入固定增益、扩展 ducking 和 RMS 检查，但人工听审仍发现关键错误：caption 声称存在 music/applause，实际没有；部分样本实际是重复 speech；vocal 仍过轻。其 source pool 仍主要复用两条 MOSS speech demo 和一条 `game.mp3`。这说明失败不再适合用 v5、v6 参数迭代解释。

下一步只回答一个可证伪问题：**在保持同一 renderer 和 active-RMS 门禁不变时，把 SFX/ambience 从单一 ESC-50 扩展为 ESC-50 + FSD50K，是否能通过源级审核、跨库均衡、可听性和 60 条全量混音听审？**

在这个问题通过前：

- 不开始 B3 或模型训练；
- 不用训练指标反推数据错误；
- 不引入 LLM 决定标签真值；
- 不继续修改 gain、ducking 或 caption prompt。

## 2. 音源扩充方案

| 阶段 | 音源角色 | 数据集 | 本阶段是否运行 | 监督和风险控制 |
|---|---|---|---|---|
| D0 | speech | LibriSpeech | 是 | 官方逐字 transcript；speaker/chapter 防泄漏 |
| D0 | SFX/ambience | ESC-50 | 是 | 类别标签；Freesound clip ID 防泄漏 |
| D1 | SFX/ambience | FSD50K | **本轮新增并运行** | 只保留“至少两名标注者判断 present and predominant”的单一主导类；排除 speech/music；按 uploader 分组；逐 clip 许可过滤 |
| D1 | music/vocal | MUSDB18-HQ | 代码已就绪，待 speech+SFX pilot 通过后运行 | music 用 drums+bass+other 的无损浮点 stem 和；vocal 单独 stem；按歌曲和艺术家防泄漏；不伪造 lyrics |
| D2 | room/noise | OpenSLR RIRS_NOISES | 暂不运行 | 只用于后续 corruption/RIR，不作为语义事件库 |

FSD50K 官方发布包含 51,197 条、约 108.3 小时、200 类音频。当前严格转换规则在官方 metadata 上得到 16,622 条候选、137 个主导类别，其中约 15,051 条 SFX、1,571 条 ambience；只接受 CC0/CC-BY 时，按官方 train/val/test 分别约为 12,331/858/3,433 条。该数字是 metadata dry run 的结果，服务器上的音频探测和人工审核仍必须通过。

FSD50K 是弱标注的真实录音，不应称为完全隔离的 dry source。因此代码先使用 PP/PNP 共识筛出“source-like”片段，再要求人工试听。LLM 可以在更晚阶段提出 scene graph 组合或改写全局 caption，但不能替代波形、主导类别和时间位置的证据。

官方来源：

- FSD50K：<https://zenodo.org/records/4060432>
- MUSDB18：<https://sigsep.github.io/datasets/musdb.html>
- MUSDB18 Zenodo：<https://zenodo.org/records/1117372>
- OpenSLR RIRS_NOISES：<https://www.openslr.org/28/>

## 3. 本轮实现的代码

### 3.1 FSD50K 下载、转换和审核

- `scripts/download_fsd50k.py`：固定 Zenodo record 和发布方 MD5；支持断点下载；安全解压；FSD50K 分卷必须通过系统 `zip -s 0` 合并。
- `sceneledger-prepare-fsd50k`：读取官方 ground truth、逐 clip metadata 和 PP/PNP ratings；排除 speech/music；过滤逐 clip 许可；将上传者固定在单一 split；生成可探测、可哈希、可人工审核的 catalog。
- `text_is_verbatim=false`：FSD50K 类别描述不能冒充逐字文本。

### 3.2 多 catalog 均衡和泄漏门禁

文件数量不是采样概率。若直接对所有文件均匀采样，约 16k 条 FSD50K 会淹没 ESC-50。因此 `CatalogSetSourcePool` 先按 `sampling_weight` 选择数据集，再在数据集内部选择音源；默认 ESC-50:FSD50K 为 1:1。

组合 catalog 时会拒绝：

- 相同 stable source ID；
- 相同解码内容 fingerprint；
- 相同 source/leakage group，例如 ESC-50 和 FSD50K 指向同一 Freesound clip；
- 同一 scene 内重复 source、speaker、recording/uploader/artist group。

扩容 pilot 的 scene-plan 门禁还要求 SFX 和 ambience 均覆盖两个数据集，且每个数据集在相应 kind 中的 slot 占比为 25%--75%。数据集内部先均匀选择 primary class、再在类内随机选择 recording，避免长尾库被少数高频类支配；60 条 pilot 至少覆盖 20 个 SFX 主类和 8 个 ambience 主类。因此“代码配置了两个库但实际只用了一个库”或“文件很多但语义类别仍单一”都会自动失败。

### 3.3 MUSDB18 music/vocal 路径

项目不会替用户接受 MUSDB18 的 educational-use 条款，也不会自动下载受条款约束的音频。`scripts/download_musdb18_metadata.py` 只下载固定 commit 的官方 tracklist；用户合法取得 MUSDB18-HQ 后，`scripts/materialize_musdb18_stems.py` 在本地分块生成：

- `accompaniment.wav = drums + bass + other`；
- `vocals.wav = vocals`。

输出使用 float WAV，不在 stem 阶段 clip，从而保持 sample-accurate stem sum。vocal caption 只描述“isolated vocals + genre”，明确标记 lyrics 未转录；只有确有 transcript 的 LibriSpeech speech 才能进入 verbatim loss。

## 4. 你现在需要执行的实验

建议至少预留 80 GB 临时空间。FSD50K 完整下载和逐条解码探测耗时远高于 60 条混音，不需要 GPU，但需要稳定网络、`ffmpeg`、Info-ZIP 的 `zip` 命令和耳机。使用一个全新的运行目录。

### 4.1 环境与基线 source catalog

```bash
export REPO=/path/to/complex-audio-caption
export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/source_expansion_v1
cd "$REPO"
git pull --ff-only
python -m pip install -e '.[data,dev]'
command -v zip
python -c 'import sceneledger, pathlib; print(pathlib.Path(sceneledger.__file__).resolve())'
```

最后一条必须指向当前 checkout 的 `$REPO/src/sceneledger`。已有通过并与当前波形哈希绑定的 LibriSpeech、ESC-50 catalog 可以复用；否则先严格执行 `docs/27_real_speech_sfx_evidence_pilot.md` 的 5.2--5.3。为避免把新 `RUN_ROOT` 误当成旧制品路径，显式记录两个目录：

```bash
# 复用时改成上一轮真实的 prepared 绝对路径；重新构造时指向本轮输出。
export LIBRI_PREP=/path/to/passed/librispeech/prepared
export ESC_PREP=/path/to/passed/esc50/prepared
test -f "$LIBRI_PREP/source_audit_report.json"
test -f "$ESC_PREP/source_audit_report.json"
```

如需同时扩充 speech train bank，可另下载 `full-clean`；本轮 60 条 test pilot 仍使用官方 `test-clean`，避免同时改变两个实验变量：

```bash
python scripts/download_librispeech.py \
  --output-dir "$DATA_ROOT/librispeech_full_clean" --profile full-clean
```

### 4.2 下载并构造 FSD50K source catalog

必须使用 `full`，因为本实验需要官方 eval/test 音频；`dev` profile 不能生成完整的 test catalog。

```bash
python scripts/download_fsd50k.py \
  --output-dir "$DATA_ROOT/fsd50k" --profile full

export FSD_ROOT="$DATA_ROOT/fsd50k/FSD50K"
sceneledger-prepare-fsd50k \
  --root "$FSD_ROOT" \
  --output-dir "$RUN_ROOT/sources/fsd50k" \
  --allow-license CC0-1.0 \
  --allow-license 'CC BY 3.0' \
  --audit-per-kind 30 \
  --min-per-kind-per-split 50
```

不要只看命令退出码。检查：

```bash
python - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['RUN_ROOT']) / 'sources/fsd50k/prepared/source_catalog_report.json'
r = json.loads(p.read_text())
print('pass=', r['pass'])
for c in r['checks']:
    if not c['pass']:
        print('FAIL', c['name'], c['detail'])
PY
```

用耳机逐条打开 `$RUN_ROOT/sources/fsd50k/prepared/source_audit.csv` 中的 60 条任务（30 SFX + 30 ambience），填写 `audible_y_n`、`caption_correct_y_n`、`kind_correct_y_n`。多声源片段只有在目标类确实主导且简短 caption 被波形支持时才能标 `y`；不确定就标 `n` 并写 notes，不要为了过门禁降低标准。

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$RUN_ROOT/sources/fsd50k/prepared/source_catalog_report.json" \
  --audit-csv "$RUN_ROOT/sources/fsd50k/prepared/source_audit.csv" \
  --output "$RUN_ROOT/sources/fsd50k/prepared/source_audit_report.json" \
  --min-per-kind 30 --min-pass-rate 0.90 \
  --required-split test --min-per-kind-per-required-split 10
```

### 4.3 冻结并渲染 60 条扩容 pilot

下面假设 LibriSpeech、ESC-50 路径与上一轮一致：

```bash
export LIBRI_ROOT="$DATA_ROOT/librispeech"
export ESC_REPO="$DATA_ROOT/esc50/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
export ESC_AUDIO="$ESC_REPO/audio"

python scripts/make_real_speech_sfx_pilot_config.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$LIBRI_PREP" \
  --esc50-audio-root "$ESC_AUDIO" \
  --esc50-prepared "$ESC_PREP" \
  --fsd50k-root "$FSD_ROOT" \
  --fsd50k-prepared "$RUN_ROOT/sources/fsd50k/prepared" \
  --output "$RUN_ROOT/expanded_pilot_test.yaml"

bash scripts/run_real_speech_sfx_pilot.sh \
  "$RUN_ROOT/expanded_pilot_test.yaml" \
  "$RUN_ROOT/output" \
  expanded_pilot
```

runner 会按顺序执行 import-path 检查、scene-plan preflight、render、replay/stem/Ledger validation、mixture quality gate，并生成 60 条全量人工任务。自动门禁失败时停止，不要人工听一批已知无效数据。

### 4.4 全量听审和回传

若自动门禁通过，逐条播放 `$RUN_ROOT/output/human_audit_tasks.csv` 的 `mixture_path`；有 stem review 要求时同时播放 `stem_paths_json`。60 条全部填写，不接受空值或 `uncertain` 作为 GO 证据。

```bash
bash scripts/summarize_real_speech_sfx_pilot.sh "$RUN_ROOT/output"
```

请回传以下小文件，不要上传原始数据：

- 当前 commit、`pip_freeze.txt`、`sceneledger_import_path.txt`；
- 三个 source catalog report、完成的 source audit CSV 和 source audit report；
- `expanded_pilot_test.yaml`、`scene_plan_preflight.json`；
- `manifest.jsonl`、`mixture_quality.json`；
- 完成的 `human_audit_tasks.csv`、metadata 和 summary；
- 若失败，附 3--5 条代表性 mixture/stems 和中文听感说明。

## 5. GO / NO-GO 和后续顺序

自动 GO 必须同时满足：跨库占比、主类别覆盖、source provenance、无 source reuse、active-RMS floor、speech competitor margin、replay、stem sum、Ledger schema 和无 clipping。人工 GO 沿用 pilot 的严格标准：无 severe，overall failure 不超过 2/60，同一模板同一 criterion 不得形成系统性重复失败。

若失败，只修改报告指向的单一环节：

- FSD source audit 失败：调整 FSD 筛选/类别映射，不改 mixer；
- dataset fraction 失败：调整冻结前的 bank weight/seed，不改音频增益；
- speech/SFX 不可听：检查 source active RMS 和 post-render stem，不改 caption；
- caption 不符：修 source record，不用 LLM 猜测替换；
- timestamp/stem/replay 失败：修 renderer/activity，不扩数据规模。

只有本轮 GO 后才进入 music/vocal pilot。届时先由用户在官方页面取得 MUSDB18-HQ，再运行：

```bash
python scripts/download_musdb18_metadata.py \
  --output "$DATA_ROOT/musdb18/tracklist.csv"
python scripts/materialize_musdb18_stems.py \
  --input-root /path/to/MUSDB18-HQ \
  --output-root "$DATA_ROOT/musdb18/materialized"
sceneledger-prepare-musdb18 \
  --root "$DATA_ROOT/musdb18/materialized" \
  --tracklist "$DATA_ROOT/musdb18/tracklist.csv" \
  --output-dir "$RUN_ROOT/sources/musdb18" \
  --allow-license 'CC BY-NC-SA'
```

官方 tracklist 的 `CC BY-NC-SA` 行没有给出版本号，所以代码原样保存，不擅自推断为 4.0。若该 allowlist 不能满足每个 split 至少四首歌，先检查官方 tracklist 和 split 报告；不得无记录地把 `Restricted` 加入 allowlist。训练集规模扩展、互联网真实混合 D1/D2、LLM scene proposal、RIR/noise robustness 都排在两个 source-level pilot 通过之后。
