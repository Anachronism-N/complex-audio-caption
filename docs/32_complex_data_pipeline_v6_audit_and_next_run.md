# 复杂数据管线 v1：v6 审计、六源锚点与下一轮执行

更新日期：2026-08-15

## 1. 结论先行

当前不应继续解释 B3 的模型效果，也不应马上做 recaption。仓库中的 `real_mix_v6` 不满足复杂音频训练集的基本定义，必须先被替换为一个可审计的数据锚点。

本轮新增的第一个锚点固定为：

```text
12--18 s scene
├── speech A：完整 utterance，保留逐字 transcript
├── speech B：不同 speaker，和 A 有重叠
├── ambience：从 0 s 开始，必要时循环或随机截取
├── SFX 1：前段
├── SFX 2：中段
└── SFX 3：后段
```

每条 scene 因而有 6 个可追溯 source，目标不是先追求“自然语言漂亮”，而是证明：source 身份真实、时序可重放、重叠足够、声音可听、组合不再退化为两个顺次长块。recaption 明确延期；原始 transcript/类别标签仍是唯一语义监督来源。

## 2. 为什么旧 v6/B3 结果不能作为当前锚点

对 `data/derived/real_mix_v6/manifest_compat.jsonl` 运行新增的 manifest-only 审计后，得到：

| 指标 | v6 实测 | 诊断要求 | 结论 |
|---|---:|---:|---|
| scene 数 | 200 | >= 50 | 通过 |
| 平均 source 数 | 2.755 | >= 3.0 | 失败 |
| 平均 event 数 | 2.755 | >= 3.0 | 失败 |
| 严格复杂 scene 比例 | 0.000 | >= 0.30 | 失败 |
| 多 voice scene 比例 | 0.000 | 当前诊断未设下限 | 与任务不符 |
| provenance 完整率 | 0.000 | 1.000 | 失败 |
| 2-source / 3-source 数 | 49 / 151 | — | 复杂度上限过低 |

这里的“严格复杂”要求至少 4 source、4 event、15% 时间有重叠、峰值至少 2 个同时 track。v6 虽然平均 overlap ratio 为 0.503，但只有 2--3 个 source，因此 **overlap 高不等于 scene 复杂**。

此外，`convert_v6_manifest.py` 把真实路径抹成 `real:speech`、`real:sfx`、`real:music`，没有保留 `source_group`、原文件 hash、数据集标签和 stem。因此：

- 无法证明 train/val/test 按原始 speaker/recording/song 隔离；
- 无法重放每个 source 对 mixture 的贡献；
- 不能把 B3 的 `macro event F1=0.865` 解释为复杂场景泛化；
- 该结果只说明模型能够拟合这批 2--3 source 的旧目标格式。

冻结的诊断报告在 `reports/real_mix_v6_complexity_audit.json`。

## 3. 新实现到底改变了什么

### 3.1 可审计的复杂度门禁

新增 `sceneledger-audit-complexity`。它只读取 frozen manifest，不读取模型预测，统计：

- source、track、event 数；
- active ratio、overlap ratio、最大同时 track 数、时间边界变化数；
- simple、sequential-only、block-like、strict-complex 比例；
- 多 voice、speech+music+SFX、music+vocal 比例；
- 每类唯一 source group 数；
- source provenance 是否包含真实路径、dataset、group、SHA-256 和原标签。

`configs/data/complexity_profiles.yaml` 中有三种 profile：

- `diagnostic_v1`：用于拒绝旧 v6 一类数据；
- `speech_sfx_complex_v1`：本轮可执行的 120 条六源锚点；
- `full_joint_complex_future`：music/vocal 人工审核通过后才允许使用。

source audit 还会把任一 `audible/caption_correct/kind_correct=n` 的已知坏样本写入 `rejected_source_ids`。`CatalogSourcePool` 即使在总体抽检通过时也会隔离这些 ID；隔离后 source 数量或类别覆盖不足，则 scene-plan preflight 失败。也就是说，90% 抽检阈值只用于估计整个 bank 的质量，不能让已经确认错误的那 10% 继续进入混音。

### 3.2 六源 scene，而不是只增加 duration

新增模板 `multi_speaker_ambient_events`：

- 两条 speech 必须来自不同 audio path、speaker/source group；
- speech onset 约在 scene 的 8% 和 12%，形成有意的说话重叠；
- 三条 SFX 目标 onset 约在 12%、42%、70%，覆盖前中后段；
- ambience 贯穿 scene；
- 单 scene 内禁止重复 path 和非有意的 group/leakage-group 重用。

这直接针对“11 秒音频只有一个音效”或“前 5 秒一个、后 5 秒一个”的退化分布。注意，ESC-50/FSD50K 的 label 仍可能是 clip-level；renderer 通过真实波形 activity 形成 span，但人工仍需检查类别是否在目标时段可听。

### 3.3 音量、裁剪、淡入淡出与房间

混音参数变为 manifest 的一部分，而不是不可追踪的临时 DSP：

- `crop_start_sec`、`crop_duration_sec`：长 ambience/music 使用随机 excerpt；
- `fade_in_sec`、`fade_out_sec`：按 kind 分别采样，SFX/speech 使用毫秒级边界，ambience 使用较慢过渡；
- active-RMS target：speech `[-22,-20] dBFS`，SFX `[-32,-28] dBFS`，ambience `[-40,-36] dBFS`；
- 65% scene 使用一个共享 room ID/T60，15% 使用显式 echo；
- 25% 使用 2--4 dB 的轻量 ducking，且 ducking 作用到持久化 stem，能够精确消融和重放；
- master gain 同时缩放 mixture 和所有 stems，保证 stem-sum 恒等式。

当前没有加入随机 phase。原因是当前训练输入最终为 mono，任意相位/极性扰动既缺少麦克风几何依据，也可能只制造抵消伪影。真正的空间化需要明确的多通道输出、source/microphone geometry 和 measured RIR；在此之前，共享房间参数比无依据的随机 phase 更可解释。

### 3.4 music/vocal 不再随机错配

MUSDB18 一首歌的 accompaniment 和 vocal 具有相同 `source_group`。旧 sampler 会把同 group 当作 scene 内泄漏并拒绝，因此 `lyrics_over_music` 实际无法保证同歌对齐。

现在 music/vocal 模板会：

1. 先找同时含有 `music` 与 `vocal` 的 paired group；
2. 从同一首歌选择两个 stem；
3. 使用完全相同的 crop start/duration；
4. 若没有经过审核的 paired group，直接 fail closed，不再拿两首无关歌曲混合。

MUSDB18 没有歌词 transcript，因此 vocal 只监督“有人声/演唱活动”，不能生成伪歌词。该路径已有代码和测试，但不进入本轮 P1 六源锚点。

## 4. 音源扩展：不是所有数据都拿来当 dry source

机器可读计划见 `configs/data/dataset_expansion_plan.yaml`。建议顺序如下。

### P1：现在执行

- LibriSpeech：两个不同 speaker、完整 utterance、官方 transcript；
- ESC-50：可解释的 isolated-event anchor；
- FSD50K：只保留高一致性 predominant class，增加类别和 uploader 多样性。

### P2：P1 通过后做真实分布和声学鲁棒性

- DESED 同时提供 real soundscapes、soundbank 和 Scaper synthetic soundscapes，适合作为 domestic real-scene 外部验证，不应全部当 isolated dry source（[官方仓库](https://github.com/turpaultn/DESED)）。
- OpenSLR RIRS_NOISES 提供真实/模拟 RIR 与噪声，可替换当前 synthetic RIR corruption（[官方页面](https://www.openslr.org/28/)）。
- FUSS 面向任意数量 source 的分离，发布 source、mixture、RIR/混音资源，但 source 文件没有语义标签；适合 source-count/分离诊断，不能直接产生 caption 真值（[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/fuss)，[Google Open Source 说明](https://opensource.googleblog.com/2020/04/free-universal-sound-separation.html)）。
- TAU Urban Acoustic Scenes 适合作为真实城市 ambience/domain-shift 数据，而不是 discrete SFX 库（[官方 Zenodo](https://zenodo.org/records/6337421)）。

### P3：多说话人和 music/vocal

- LibriMix 提供基于 LibriSpeech、WHAM 的 2/3-speaker mixture protocol，可作为说话重叠比较锚点；默认全量生成占用很大，应只生成预注册子集，并继续遵守底层数据许可（[官方仓库](https://github.com/JorisCos/LibriMix)）。
- MUSDB18-HQ 先完成 source audit，再运行本轮新增的 aligned pair sampler。
- MedleyDB 有 multitrack/stem 与 instrument metadata，适合后续细粒度音乐描述，但需要先完成访问和再分发许可审核（[官方站点](https://medleydb.weebly.com/)，[官方文档](https://medleydb.readthedocs.io/en/latest/example.html)）。

DNS Challenge 体量较大，只在上述门禁通过后作为大规模 noise/RIR robustness bank（[官方仓库](https://github.com/microsoft/DNS-Challenge)）。

## 5. 服务器上下一步怎么运行

以下命令假设 LibriSpeech、ESC-50、FSD50K 均已经完成 source prepare 和人工 source audit。FSD50K 必须使用包含官方 eval/test 音频的 `full` profile；详细下载与审核步骤继续使用 `docs/29_expanded_source_bank_protocol.md`。

```bash
cd /path/to/complex-audio-caption
python -m pip install -e '.[dev]'

export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/complex_speech_sfx_v1
export LIBRI_ROOT="$DATA_ROOT/librispeech"
export ESC_AUDIO="$DATA_ROOT/esc50/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio"
export FSD_ROOT="$DATA_ROOT/fsd50k/FSD50K"
export LIBRI_PREP=/path/to/passed/librispeech/prepared
export ESC_PREP=/path/to/passed/esc50/prepared
export FSD_PREP=/path/to/passed/fsd50k/prepared

python scripts/make_complex_speech_sfx_config.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$LIBRI_PREP" \
  --esc50-audio-root "$ESC_AUDIO" \
  --esc50-prepared "$ESC_PREP" \
  --fsd50k-root "$FSD_ROOT" \
  --fsd50k-prepared "$FSD_PREP" \
  --sample-count 120 \
  --output "$RUN_ROOT/complex_test.yaml"

# 先审核 builder 同时生成的 120 条 label-level rule recipe：
# $RUN_ROOT/complex_test.rule_recipe_review.csv
# 每行填写 plausible_y_n、label_compatible_y_n 和必要的 notes。
# 任一 recipe 被拒绝或存在空白时，运行脚本会在渲染前 fail closed；
# 修正规则或 recipe plan 后重新审核，不把已知不合理 recipe 留在数据中。

bash scripts/run_complex_speech_sfx_pilot.sh \
  "$RUN_ROOT/complex_test.yaml" \
  "$RUN_ROOT/run"
```

builder 会同时输出 frozen inventory、120 条 keyword-rule recipes 和 recipe review CSV。运行脚本的顺序固定为：recipe 人工门禁 → import 路径校验 → scene-plan preflight → render/replay/stem-sum → waveform quality → complexity audit → 60 条 mixture 人工 review 任务。任一步失败即停止，禁止靠降低门槛继续训练。

## 6. 人工审核应回答什么

`$RUN_ROOT/run/human_audit_tasks.csv` 至少试听 60 条，建议随机顺序、戴耳机完成。每条检查：

1. 两个说话人是否都可听、是否确实重叠、transcript 是否各自成立；
2. 三个 SFX 是否都存在，类别描述是否由音频支持；
3. SFX 是否覆盖前中后，而非全挤在同一位置；
4. ambience 是否自然连续，loop/crop/fade 是否出现周期接缝；
5. room/echo 是否自然，是否把离散事件拖成长伪事件；
6. 任一 source 是否被遮蔽到不可辨识；
7. 整体是否像合理场景，而不是“六条互不相关声音机械叠加”。

自动门禁回答的是“数据结构和声学证据存在”；第 7 项的因果/常识合理性必须由人工确认。只有 rule-based 六源锚点通过后，才比较 LLM recipe：LLM 只能从冻结 inventory 选择哪些 label 共现，不能改写 source、timestamp、transcript、stem 或 Ledger。

## 7. Go / No-Go 标准

进入模型训练必须同时满足：

- `scene_plan_preflight.json: pass=true`；
- `recipe_review_report.json: pass=true`，120 条 rule recipe 填写完整且通过率 = 100%；
- `mixture_quality.json: pass=true`；
- `complexity_audit.json: pass=true`；
- 120 条中 strict-complex 比例 >= 60%，本模板预期接近 100%；
- provenance 完整率 = 100%；
- 唯一 speaker group >= 30、SFX group >= 80、ambience group >= 30；
- 60 条人工 mixture review 整体通过率 >= 90%；
- 单 source 可听率 >= 95%；
- 不允许系统性的 loop click、截断 transcript、错配 source label 或不自然 echo。

若失败，先按 failure category 修 renderer/source policy；不要训练，不要启动 recaption，也不要继续调模型 loss。

## 8. recaption 延期后的边界

本阶段保留：

- speech：LibriSpeech 原 transcript；
- ESC-50：官方 class label 的保守模板文本；
- FSD50K：高一致性 predominant class 的保守模板文本；
- music/vocal：原 stem metadata，不生成歌词。

recaption 后续可以作为单独实验，前提是先建立 source-level 人工集并分别评估 semantic precision、unsupported attribute rate 和 hallucination rate。它不能覆盖原始 caption 字段，也不能改变 source/event 时间真值。这样即使 recaption 失败，当前复杂数据锚点仍然可复现、可回退。
