# 下一步实验：可追溯的 LibriSpeech + ESC-50 证据闭环

更新日期：2026-08-13

## 1. 结论先行

当前不应训练 B3，也不应继续调 loss、EQ、压缩器或 MOSS prompt。下一步唯一主实验是一个 **30 条、test-only、全量人工听审的 speech + SFX/ambience pilot**。它只回答一个前置问题：在每个干声源都可追溯、文本监督可靠、源文件不被替换、非连续事件不被裁切的条件下，现有 sampler → renderer → 0.1 s Ledger 流程能否生成与实际听感一致的数据。

只有该实验通过，才构造 train/val 并开始 baseline 训练。未通过时，根据失败字段定位到 source、mixer 或 timestamp/VAD，不能用训练结果反推原因。

## 2. 拉取后的远端代码分析

本地分支已包含远端 `main` 截至 `cf61605` 的全部提交，包括 v3/v3b mixer、200 条 v3b manifest、10 条 review 任务与人工反馈。详细审计见 `docs/28_v3b_audit_and_active_rms_gate.md`。这些工作证明“真实波形优于占位合成波形”的方向正确，但 v1/v2/v3/v3b 目前都只能当诊断样本：

1. `real_mix_v2` 共 200 个 scene、538 个 source slot，其中 SFX 463、music 49、speech 26；538/538 都没有稳定源 ID、源文件哈希或 source group。`real_mix` 的 498 个 slot 也是如此。
2. 无法检查同一原始录音是否跨 split 泄漏，无法统计真实 source reuse，也无法证明人工听审和训练读取的是同一个波形。
3. v2 的 music/speech 来自很少的 MOSS demo 文件；`license_status="CC"` 不是可审计的精确许可证。
4. caption 是 MOSS 自动候选并被截断，confidence 却统一硬编码，数值没有校准依据。
5. 人工反馈已经指出 speech 清晰度、SFX 遮蔽和“带人声歌曲冒充纯音乐”的问题。此时继续调固定 gain 会把 source、标签和 renderer 缺陷混在一起。
6. v3 的 200 次循环只有前三行位于循环体内，实际渲染逻辑错误地在循环外执行；v3b 虽有 200/200 行可解析 JSONL，却不是仓库当前可 deterministic replay 的 canonical `ManifestEntry`，不能通过 replay、stem-sum 和 Ledger gate。

所以本轮不对 v2 做第 3 次参数微调，而改用有正式 transcript 和许可证的 LibriSpeech speech anchor；ESC-50 作为已知类别的真实 SFX anchor。

## 3. 实验假设与边界

### 3.1 假设

- LibriSpeech 的完整 utterance transcript 与一个 speech event 一一对应；VAD 停顿只能形成同一 event 的多个 span，不能复制出多个完整 transcript。
- ESC-50 class label 只作为弱但可溯源的 SFX/ambience 语义，不冒充丰富自然语言 caption。
- placement 可从 stem 自动产生 0.1 s 网格的可重放 activity span。
- 在无 RIR、无 echo、speech 相对更响的简单混合中，人工应能听清 speech、核对 transcript、听见 SFX，并确认 span 基本覆盖实际事件。

### 3.2 不证明什么

本实验不证明 music、lyrics、多说话人、真实混响/回声、全局声学场景或 0.1 s 边界误差已解决，也不测模型。它先隔离验证数据管线。

### 3.3 与 MOSS caption anchor 的关系

`docs/26_esc50_real_audio_anchor.md` 的 50 类 × 2 prompt 实验仍用于判断“能否自动扩写 rich SFX caption”。本 pilot 先用 dataset label 回答更基础的波形、类别、speech transcript、时间和 stem 一致性；两者不互相阻塞。

## 4. 本轮实现

### 4.1 可靠 speech source

- `scripts/download_librispeech.py`：下载 OpenSLR 的 Mini LibriSpeech train/dev 与官方 `test-clean`，校验发布方 MD5，拒绝路径逃逸和链接文件。
- `sceneledger-prepare-librispeech`：逐 utterance 保留原始 transcript、speaker、chapter、官方 split、CC BY 4.0、来源 URL；检查解码、响度、裁切、文件 SHA-256、内容 fingerprint、caption 多样性和 speaker 泄漏。
- speaker 是 `source_group`，chapter 是附加 `leakage_group`；官方 split 如出现说话人冲突会直接失败。

### 4.2 审核后不可偷换源文件

prepared catalog 与人工 source audit 通过 SHA-256 绑定；审核表的 source ID、split、kind、路径、caption 和 identity 另有 canonical task hash，审核者只能填写答案与 notes，换样本或改任务文本都会失败。`CatalogSourcePool` 在某源首次进入 plan/render 时重新计算原音频 SHA-256；文件缺失、路径越出 `audio_root`、catalog 改动或波形被替换都会停止。manifest 携带 stable catalog ID、group、dataset、精确 license、annotation origin、source file SHA-256 和 duration。

### 4.3 不允许裁切 transcript-bearing source

sampler 按 scene duration 过滤 speech/vocal/SFX，只选能完整放入场景的非连续源并限制 onset；preflight 再检查 `noncontinuous_sources_fit_scene`。不会出现“音频只有半句话，target 却含完整 transcript”。music/ambience 才允许 loop/crop。

### 4.4 Ledger 修正

一个 LibriSpeech 文件只生成一个 `<speech>` event，VAD 停顿变成该 event 的多个 spans。只有 `human` 或 `dataset` transcript 可标 `verbatim=true`；模型猜测的 speech/lyrics 不能冒充逐字文本。semantic confidence 由 annotation origin 映射，不再统一硬编码。

### 4.5 混合与门禁

固定 seed 生成 30 条：10 条 `speech_with_sfx`、20 条 `speech_ambience_sfx`。时长 6–10 s、16 kHz、无 RIR/echo/ducking。不同数据集的原始振幅不可直接比较，因此不再对 speech、SFX、ambience 叠加固定 gain offset；source catalog 先测 active RMS，sampler 再将三类源分别归一化到 -22~-20、-31~-27、-38~-34 dBFS，且拒绝绝对值超过 24 dB 的异常增益。这不是最终分布，而是先隔离可懂度和监督正确性。

渲染前要求：speech 至少 15 个唯一 utterance/8 个 speaker，SFX 至少 15 个唯一源/组，ambience 至少 8 个；任一源在其 kind 内复用率不超过 15%；provenance/hash/duration/全段 RMS/active RMS 完整；非连续源不被裁切；两个模板和平均 source 数达标。

渲染后直接读取写盘的 PCM stem，要求全部可解码、speech/SFX/ambience active RMS 分别不低于 -28/-38/-45 dBFS，且有竞争声重叠时 speech active RMS 至少高 3 dB。至少 80% speech scene 必须能测得重叠 margin，低 margin scene 不超过 10%。master clipping guard 对 mixture、dry mixture 和各 stem 使用同一缩放，避免 audit 读到的 stem 与实际混入贡献不一致。

### 4.6 全量人工审核

30/30 条必须听。CSV 新增 `caption_accuracy`、`speech_intelligibility`、`speech_transcript_accuracy`，并保留 audibility、timestamp、overlap、长静音、clipping、stem-mixture consistency、severity 与 overall。带 speech 的样本不能把 speech 字段写成 `not_required`；空值、`uncertain`、任务字段改动或 summary 后 CSV 改动都会 fail closed。

## 5. 你在服务器上需要做什么

这是当前唯一建议顺序；不要同时启动 B3 训练。

### 5.1 环境

```bash
export REPO=/path/to/complex-audio-caption
export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/real_speech_sfx_pilot_v2_active_rms
cd "$REPO"
git pull --ff-only
python -m pip install -e '.[data,dev]'
python -c 'import sceneledger, pathlib; print(pathlib.Path(sceneledger.__file__).resolve())'
```

最后一条必须位于 `$REPO/src/sceneledger`；否则先清理旧 editable install。

必须使用新的 `RUN_ROOT`，并用当前代码重新执行两个 source prepare；旧 catalog 没有 `active_rms_dbfs`，配置生成器会拒绝复用。

本 pilot 不需要 GPU。LibriSpeech 三个归档的发布大小约为 332 MB、126 MB 和 346 MB；加上 ESC-50、解压文件、30 条 mixture/stems 与环境记录，建议至少预留 5 GB。人工工作量是 ESC source 20 条 + LibriSpeech source 30 条 + mixture 30 条，共 80 次短音频判断。

### 5.2 ESC-50 source gate

```bash
python scripts/download_esc50.py --output-dir "$DATA_ROOT/esc50"
export ESC_REPO="$DATA_ROOT/esc50/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
export ESC_AUDIO="$ESC_REPO/audio"
sceneledger-prepare-esc50 \
  --metadata "$ESC_REPO/meta/esc50.csv" \
  --audio-root "$ESC_AUDIO" \
  --output-dir "$RUN_ROOT/sources/esc50"
```

戴耳机逐条审核 `$RUN_ROOT/sources/esc50/prepared/source_audit.csv`，填写 `audible_y_n`、`caption_correct_y_n`、`kind_correct_y_n` 为 `y/n`（默认 ambience 10 + SFX 10）：

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$RUN_ROOT/sources/esc50/prepared/source_catalog_report.json" \
  --audit-csv "$RUN_ROOT/sources/esc50/prepared/source_audit.csv" \
  --output "$RUN_ROOT/sources/esc50/prepared/source_audit_report.json" \
  --min-per-kind 10 --min-pass-rate 0.90 \
  --required-split test --min-per-kind-per-required-split 3
```

失败时检查真实 clip 与类别，不要直接降阈值。

### 5.3 LibriSpeech source gate

```bash
export LIBRI_ROOT="$DATA_ROOT/librispeech"
python scripts/download_librispeech.py --output-dir "$LIBRI_ROOT" --profile pilot
sceneledger-prepare-librispeech \
  --root "$LIBRI_ROOT" \
  --output-dir "$RUN_ROOT/sources/librispeech" \
  --subset train-clean-5 --subset dev-clean-2 --subset test-clean \
  --max-per-speaker 10 --audit-per-kind 30
```

戴耳机审核 `$RUN_ROOT/sources/librispeech/prepared/source_audit.csv` 的 30 条 speech。`caption` 是官方 transcript，必须实际听音频：

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$RUN_ROOT/sources/librispeech/prepared/source_catalog_report.json" \
  --audit-csv "$RUN_ROOT/sources/librispeech/prepared/source_audit.csv" \
  --output "$RUN_ROOT/sources/librispeech/prepared/source_audit_report.json" \
  --min-per-kind 30 --min-pass-rate 0.90 \
  --required-split test --min-per-kind-per-required-split 10
```

### 5.4 生成服务器配置并运行

```bash
python scripts/make_real_speech_sfx_pilot_config.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$RUN_ROOT/sources/librispeech/prepared" \
  --esc50-audio-root "$ESC_AUDIO" \
  --esc50-prepared "$RUN_ROOT/sources/esc50/prepared" \
  --output "$RUN_ROOT/pilot_test.yaml"

cd "$REPO"
bash scripts/run_real_speech_sfx_pilot.sh \
  "$RUN_ROOT/pilot_test.yaml" "$RUN_ROOT/output"
```

配置脚本要求两个 source audit 已通过且拒绝覆盖已有配置。runner 保存 commit/环境，依次执行 scene preflight、render、deterministic replay、stem sum、Ledger 和 mixture quality gate；任一步非 0 都应停止并保留报告。

### 5.5 你必须完成的 30 条混音听审

打开 `$RUN_ROOT/output/human_audit_tasks.csv`，逐条播放 `mixture_path`；`stem_review_required=yes` 时也播放 `stem_paths_json`。普通检查填 `pass/fail/uncertain`；不需要的 overlap/stem 填 `not_required`；speech 两项全部需要；长静音/clipping 填 `absent/present/uncertain`；severity 填 `none/minor/severe`。听不清就填 `uncertain`，不要猜。

```bash
bash scripts/summarize_real_speech_sfx_pilot.sh "$RUN_ROOT/output"
```

## 6. GO / NO-GO

GO 要求：所有自动 gate 通过；30 行完整且无 `uncertain`；severe 为 0；overall failure 不超过 2/30；同一模板同一 criterion 不出现 2 次及以上失败；所有哈希保持不变。

GO 后先用同一 scene plan 生成 clean 与单一 corruption 配对版，依次只加 RIR、echo、噪声；随后构造 source-disjoint train/val/test 和约 300 条 real-mix anchor，再训练 TAC/MOSS baseline。music/vocal 必须另做许可证、歌曲/演唱者 group split 和 source-level caption/lyrics audit。

NO-GO 时按字段只修一个环节：`all_sources_have_provenance` 查 source catalog；`stem_rms_floor_violation_fraction` 查 active-RMS 归一化、fade 或 clipping master gain；`speech_overlap_measured_fraction` 查 onset/activity；`speech_competitor_margin_violation_fraction` 查相对响度；transcript 不符查源映射和裁切；timestamp 错误查 VAD/fade/activity；stem/replay 错误修 renderer。不要扩大样本或开始训练。

## 7. 需要返回的最小实验包

- commit、Python、pip freeze；
- 两个 source catalog report、完成的 source audit CSV 和 source audit report；
- `pilot_test.yaml`、`scene_plan_preflight.json`；
- test manifest、`mixture_quality.json`；
- 完成的 mixture audit CSV、metadata、`human_audit_summary.json`；
- 若失败，附 3–5 条失败 mixture/stems 与听感说明。

不必上传原始 LibriSpeech/ESC-50。[OpenSLR 12](https://www.openslr.org/12/) 说明 LibriSpeech 约 1,000 小时、16 kHz、CC BY 4.0，并提供 `test-clean`；Mini train/dev 来自 [OpenSLR 31](https://www.openslr.org/31/)。下载脚本固定发布方 [OpenSLR 12 MD5](https://www.openslr.org/resources/12/md5sum.txt) 及 Mini 归档校验值。ESC-50 使用固定 commit，并分别记录完整集 CC BY-NC 3.0 与 ESC-10 的 CC BY 3.0。
