# 音源银行冻结方案与逐阶段实验指引

更新日期：2026-08-14

## 1. 本轮冻结的结论

当前失败首先是音源定义和音源质量问题，不是模型容量或 mixer 超参数问题。`game.mp3` 被当作 music，但源 caption 与人工听感表明其中可能包含 speech/vocal；十几秒样本全程只有一个 SFX，则无法监督多声源重叠、事件先后和细粒度时间定位。因此在 source bank 通过前，不再启动 B3 训练，也不再通过 v6/v7 调 gain 来解释结果。

我们把所有输入严格划成六种 bank：

1. `speech`：存在逐字 transcript、speaker ID 和 language；不得夹带音乐或第二名说话者。
2. `vocal/lyrics`：歌唱声。必须区分 `lyrics_verbatim=true` 的精确歌词和只证明“有人声”的弱监督。
3. `music`：一段语义上连续的音乐作品或伴奏是一条 track；内部钢琴、鼓、贝斯作为属性，不默认拆成多个语义 track。
4. `sfx`：有相对明确起止的离散事件，例如关门、狗叫、玻璃破碎。
5. `ambience`：持续背景，例如雨声、街道、人群底噪。
6. `corruption`：RIR、回声、设备噪声和加性噪声；它们是声学干预，不是 caption 中的语义事件。

互联网爬取的 Bilibili/Instagram/TikTok 音视频归入 `real_domain`，只能在权利、隐私和伪标签审核后用于私有域适配或真实评测，不能直接当成可追溯 dry source。

上述决策已编码在 `configs/data/source_bank_policy.yaml`。任何实验先运行策略验证器；禁止用途、许可或监督强度不匹配时，命令必须失败。

## 2. 为什么选择这些音源

| 角色 | 首选锚点 | 扩展源 | 可用监督 | 现在是否可运行 |
|---|---|---|---|---|
| speech | LibriSpeech | Common Voice | 精确逐字文本、speaker、clip 边界 | 是 |
| sfx/ambience | ESC-50 | FSD50K PP 共识子集 | 主导类别、clip 边界 | 是 |
| controlled music | Slakh2100-redux | 暂无 | instrument set、完整 mix/stems | 是 |
| real music/vocal | MUSDB18-HQ | 后续 MedleyDB | accompaniment/vocal stem；无精确歌词 | 代码可运行，需用户接受数据条款 |
| exact Mandarin lyrics | M4Singer/OpenCpop 候选 | 待定 | 歌词与对齐 | **许可审核前禁止运行** |
| corruption | RIRS_NOISES 候选 | 自有 RIR/noise | RIR/noise provenance | 尚未进入当前语义 pilot |

选择逻辑不是“数据越大越好”：

- LibriSpeech 是稳定的英文 speech 锚点；Common Voice 用于说话人、语言和录音设备多样性，但只能由用户从 Mozilla Data Collective 获取，仓库不提供镜像下载。
- ESC-50 类别清楚但规模小；FSD50K 更真实、多样，但不是隔离 stem，所以只使用至少两名标注者确认 `present and predominant` 的单一主导非 speech/music 类，并继续人工听审。
- Slakh 提供可控多乐器音乐及精确 stem。原始划分存在重复 MIDI 跨 split 泄漏，代码只接受官方修正后的 `Slakh2100-redux` 或 `Slakh2100-split2`，推荐 redux。
- MUSDB 的 `vocals.wav` 只支持 `vocal_presence`，没有歌词文本；任何 `<lys>` 逐字 loss 都不得读取 MUSDB caption。
- M4Singer/OpenCpop 在写入书面许可审核结果前保持 `gated`，不能因为论文需要歌词就把 `UNVERIFIED` 当成许可。

官方入口：

- LibriSpeech：<https://www.openslr.org/12/>
- Common Voice：<https://commonvoice.mozilla.org/en/datasets>
- Common Voice 条款：<https://commonvoice.mozilla.org/terms>
- ESC-50：<https://github.com/karolpiczak/ESC-50>
- FSD50K：<https://zenodo.org/records/4060432>
- Slakh2100-redux：<https://zenodo.org/records/4599666>
- MUSDB18：<https://sigsep.github.io/datasets/musdb.html>

## 3. 本轮代码提供什么

### 3.1 机器可检查的策略

```bash
sceneledger-validate-source-policy \
  --policy configs/data/source_bank_policy.yaml \
  --profile d0_anchor_research \
  --output /tmp/d0_policy_report.json
```

策略同时检查：dataset 是否存在、profile 是否声明可运行、角色是否匹配、逐 clip 许可是否在 allowlist、监督 claim 是否由数据支持，以及 corruption/real-domain 数据是否被误放进语义 bank。

准备好 catalog 后必须将实际制品绑定到策略，而不是只验证 YAML：

```bash
sceneledger-validate-source-policy \
  --policy configs/data/source_bank_policy.yaml \
  --profile d0_anchor_research \
  --catalog librispeech="$RUN_ROOT/sources/librispeech/prepared/test.jsonl" \
  --catalog esc50="$RUN_ROOT/sources/esc50/prepared/test.jsonl" \
  --require-catalogs \
  --output "$RUN_ROOT/d0_policy_with_catalogs.json"
```

若 speech 中 `text_is_verbatim=false`、MUSDB vocal 被声明为精确歌词、license 超出 profile allowlist，报告均为 `pass=false`。

### 3.2 Common Voice 本地导入器

导入器读取官方 `train.tsv`、`dev.tsv`、`test.tsv` 和 `clips/`：

- 默认只留 `up_votes>=2` 且 `down_votes=0`；
- 将 `client_id` 哈希后再写入 source catalog；
- 检测同一 speaker 是否跨 train/val/test；
- 拒绝路径逃逸、缺失 clip、重复 clip 和 locale 不一致；
- 只有 TSV 的原句才标记 `text_is_verbatim=true`。

仓库故意没有 Common Voice 下载脚本，因为 Mozilla 要求 Common Voice 数据只通过 MDC 访问，不应由本项目镜像。

### 3.3 Slakh2100-redux 下载与导入器

`scripts/download_slakh2100.py` 固定 Zenodo record、文件名和发布方 MD5，支持 105 GB 文件断点续传、安全解压及显式删除 archive。导入器读取每个 `metadata.yaml` 的 MIDI UUID 和 instrument classes：

- 原始 `Slakh2100-orig` 会被拒绝；
- MIDI UUID 跨 split 会被拒绝；
- 默认排除 `voice/vocal/choir/aahs/oohs` 类合成音色，避免 music bank 暗含类人 vocal；
- 整首 `mix.flac` 是一条 music source，instrument classes 是属性；
- caption 只声称可由 metadata 证明的 synthetic instrumental music 和 instrument set，不猜 genre、情绪或歌词。

## 4. 严格的实验顺序

每阶段只改变一个变量。上一阶段未通过时，下一阶段不运行。

### P0：环境、策略和已有制品检查（约 20 分钟）

```bash
export REPO=/path/to/complex-audio-caption
export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/source_bank_freeze_v1
cd "$REPO"
git pull --ff-only
python -m pip install -e '.[data,dev]'
python -m pytest -q

sceneledger-validate-source-policy \
  --policy configs/data/source_bank_policy.yaml \
  --profile d0_anchor_research \
  --output "$RUN_ROOT/p0_d0_policy.json"
```

验收：测试全过；策略报告 `pass=true`；Python import 必须指向当前 checkout。失败则停止。

### P1：先完成 speech + SFX/ambience 数据锚点（本轮首要实验，6--12 小时）

本阶段保持 LibriSpeech、renderer、active-RMS gate 不变，只把 SFX/ambience 从 ESC-50 扩为 ESC-50 + FSD50K。完整命令见 `docs/29_expanded_source_bank_protocol.md` 第 4 节。必须完成：

1. FSD50K checksum 下载、转换与 source-level 自动 gate；
2. 耳机人工审核 30 条 SFX + 30 条 ambience；
3. 冻结 60 个 scene plan，确认 ESC-50/FSD50K 在每种角色占比均为 25%--75%；
4. 渲染、replay、stem-sum、Ledger、active-RMS 和 clipping gate；
5. 人工听完 60 条 mixture，不允许空值代替审核。

P1 GO：source audit 三项通过率都不低于 90%；60 条 mixture 无 severe failure，overall failure 不超过 2 条；不存在系统性 caption 缺声、重复 speech、声称 music 但不可听等错误。

P1 NO-GO：只修改失败指向的环节。source caption 错就修 source filter；stem 不可听就查 active RMS；时间/replay 错就修 renderer。不得同时加入 Common Voice、Slakh 或训练模型。

### P2：speech 多样性单变量扩展（P1 GO 后，约 4--8 小时）

在 Mozilla Data Collective 手工选择具体 release 和 locale，推荐先分别下载英文与普通话，不要一次加入十种语言。假设解压后的 locale root 含 `train.tsv/dev.tsv/test.tsv/clips/`：

```bash
sceneledger-prepare-common-voice \
  --root "$DATA_ROOT/common_voice/cv-corpus-XX/en" \
  --release cv-corpus-XX \
  --locale en \
  --output-dir "$RUN_ROOT/sources/common_voice_en" \
  --min-up-votes 2 --max-down-votes 0 \
  --max-per-speaker 50 \
  --audit-per-kind 30 --min-per-split 20
```

先听审 30 条 source，再制作只改变 speech bank 的 60 条 matched scene plan。比较 LibriSpeech-only 与 LibriSpeech+CommonVoice；SFX、seed、slot 数、SNR、renderer 全部相同。主要观察 transcript 正确率、speech 可懂度、说话人/设备多样性和 source leakage，不看模型训练 loss。

Common Voice 的原始包和音频不得提交到本仓库或第三方对象存储；只回传 catalog report、完成的审核表、统计和可复现命令。

### P3：controlled music 单变量扩展（P1 GO 后可独立运行，约 6--12 小时加下载时间）

Slakh archive 约 105 GB，解压后接近 500 GB。先确认至少 650 GB 可用空间：

```bash
python scripts/download_slakh2100.py \
  --output-dir "$DATA_ROOT/slakh2100"

sceneledger-prepare-slakh \
  --root "$DATA_ROOT/slakh2100/slakh2100_flac_redux" \
  --split-variant Slakh2100-redux \
  --output-dir "$RUN_ROOT/sources/slakh2100_redux" \
  --min-instrument-classes 4 \
  --audit-per-kind 30 --min-per-split 20
```

如果脚本打印的 `root=` 不同，以实际值为准。不要添加 `--allow-voice-like-instruments`，除非另开消融实验并人工证明这些音色不会被当成歌声。

P3 的第一轮只构造 `speech + music` 和 `music + sfx`，每类 30 条，不加入 MUSDB vocal。验证模型之外的数据事实：music 确实可听、instrument-set caption 不虚构、一个 composition 被表示为一个 music track、source group/MIDI UUID 无跨 split 泄漏。

### P4：real music + weak vocal（P3 GO 后）

用户从 MUSDB 官方入口合法取得 MUSDB18-HQ 后：

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
  --allow-license 'CC BY-NC-SA' \
  --allow-license 'CC BY-NC-SA 3.0'
```

这一阶段只学习 `music`、`vocal_presence`、stem timing 和非逐字 vocal 描述。不得产生 `<lys>具体歌词</lys>` 监督。精确歌词必须等待 P5 的许可和对齐审核。

### P5：精确中文歌词（当前阻塞）

在以下材料齐全前不写下载器、不混音、不训练：数据集精确版本、完整 license 文本、是否允许训练与派生标注、是否允许论文复现实验、歌曲/歌手 split 规则、歌词与音素/音符对齐误差抽检。审核完成后将 `source_bank_policy.yaml` 中相应 dataset 的 `status`、`licenses` 与 profile `runnable` 通过单独 PR 修改；不得直接编辑实验输出绕过 gate。

## 5. 需要人工做什么

人工审核不能由自动 caption model 或 LLM 替代。每个新 bank 至少分层随机听 30 条/角色，填写：

- 音频可解码且非静音；
- caption/逐字文本是否由波形支持；
- kind 是否正确；
- 是否夹带未标注 speech、vocal、music 或第二事件；
- 对 music/vocal，是否错误声称歌词、风格或情绪。

混音阶段必须同时播放 mixture 和各 stem。仅听 mixture 无法判断“模型漏检”究竟是 source 不存在、stem 太轻、被遮蔽还是 caption 虚构。

## 6. 回传清单与下一步决策

每阶段回传小型制品，不回传受限原始音频：

- Git commit、环境 freeze 和 import path；
- `source_bank_policy_report.json`；
- 每个 source catalog report、完成的 audit CSV 与 audit report；
- 冻结 scene plan、render manifest、mixture quality report；
- 完成的 mixture audit 和 summary；
- 失败时附 3--5 个代表性 mixture/stems（仅在许可允许时）及中文听感。

只有 P1 数据锚点 GO 后才讨论模型实验；只有 P3/P4 GO 后才把 `<music>/<lys>` 加入联合训练。这样每次结果都有明确锚点：失败发生在哪个 bank、哪种监督、哪个 renderer gate 或哪个模型能力，而不是把所有变量同时改动后猜原因。
