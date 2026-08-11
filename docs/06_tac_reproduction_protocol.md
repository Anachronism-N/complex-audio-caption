# TAC-style 基线复现协议

> **路线状态（2026-08-09）：暂停。** 本文保留为 TAG 2021 锚点通过后的 TAC
> paper-spec 对照方案；当前第一优先级与验收标准见
> [Anchor-first TAG 复现协议](13_anchor_first_tag_reproduction.md)。

> 目标：得到一个可审计、可重复、可被 SceneLedger 公平超过的 timestamp-token baseline。由于 TAC 未公开代码、checkpoint 和其 licensed single-source corpus，本协议明确称为 **TAC paper-spec reimplementation**，不声称 exact reproduction。

## 1. 复现边界

### 已由论文公开、应尽量保持一致

- Qwen2-Audio 类 LALM + 冻结 backbone + linear-layer LoRA；本项目主实现把 backbone 替换为 MOSS-Audio-4B，并额外保留 Qwen2-Audio sanity run；
- Dynamic Acoustic Mixer、scene templates、RMS activity、随机 merge/activity/resolution；
- `keyword/brief/detailed` 三种 style；
- type prefix、按 onset 排序、多 spans；
- atomic timestamp tokens 和 timestamp-weighted CE；
- 5k optimization steps、global effective batch 32、LoRA rank 128、alpha 256、peak LR `5e-5`、1000-step warmup、cosine decay；
- TACOS test 上的 100 ms SegF1、±1 s onset collar EvtF1、FLAM Hal/conf/spec。

### 无法严格保持一致

- 未公开的 high-fidelity licensed single-source corpus；
- metadata-to-caption 清洗和 GPT-OSS/Qwen-VL instruction 扩写的完整 prompts；
- scene template 全集、增益/混响/失真分布与采样比例；
- tokenizer special-token 具体范围、optimizer 其余参数、gradient clipping、weight decay 等未披露细节；
- TAC checkpoint、随机种子和训练代码；
- 外部 Whisper 版本及 speech event 到转写窗口的后处理。

所有无法观察的变量必须写入 `configs/` 并随实验结果发布，不能为了追数字事后静默修改。

## 2. 数据版本

### R0：metric/parser smoke set

- 100–500 条程序生成样本；
- 1–3 个清晰 isolated events；
- 包含边界 0.0 s、clip end、重复 spans、完全重叠和空场景；
- 只用于单测，不汇报论文性能。

### R1：open-data baseline

建议事件来源：

- SFX/ambience：FSD50K 可再分发子集和其他许可清晰的 Freesound 数据；
- temporal real captions：TACOS train/validation；
- speech：LibriSpeech/Common Voice 等允许研究使用的带 transcript 音频；
- music/lyrics：MUSDB18 lyrics extension、Slakh2100 以及许可允许的音乐片段；
- negative/background：许可明确的 ambience、noise 和 silence。

不要把 YouTube/Bilibili/Instagram/TikTok 原始媒体混入 R1。R1 的意义是任何研究者都能按 manifest 重建。

### R2：internal web-data baseline

- 加入合规筛选后的内部互联网数据与 pseudo labels；
- 只在 R1 基线稳定后训练；
- 与 R1 分开报告，以避免数据规模掩盖方法差异；
- 论文公开 manifest 构建逻辑、平台比例和过滤统计，不公开无权再分发的媒体。

## 3. Mixer 规范

### 3.1 Scene manifest

每个 mix 必须能从 manifest 和源音频复建：

```yaml
scene_id: mix_000001
seed: 1947
duration: 12.8
template: speech_over_music_with_transient
sources:
  - source_id: music_01
    type: music
    path: ...
    onset: 0.0
    gain_db: -8.0
    rir_id: room_03
  - source_id: speech_01
    type: speech
    path: ...
    onset: 0.7
    gain_db: -2.0
    speaker_id: spk_17
  - source_id: glass_01
    type: sfx
    path: ...
    onset: 4.6
    gain_db: 1.0
conditions:
  noise_snr_db: 5.0
  echo_delay_ms: 180
  codec: opus
supervision:
  style: brief
  activity_threshold: 0.05
  merge_threshold_s: 0.25
  resolution_s: 0.1
```

保存 dry stem、processed stem、mix 与 activity mask 的 hash。训练时可在线混音，但 validation/test 必须预生成并冻结。

### 3.2 RMS activity

对每个 processed stem 计算短时 RMS，归一化到该事件的参考峰值：

$$
r_i(t)=\sqrt{\frac{1}{W}\sum_{n=t}^{t+W-1} a_i[n]^2},\qquad
m_i(t)=\mathbb{1}[r_i(t)>\delta_{act}\,r_i^{max}].
$$

建议内部 hop 为 10 ms，生成标签时再聚合到目标 resolution。对持续音乐或 noise，单纯相对峰值可能把弱段误切断；需要记录原始 activity curve，并对 music 使用较低阈值/包络平滑。该改动进入 `B2-complex`，`B2-paper-spec` 保持统一 RMS 逻辑。

### 3.3 建议采样范围

以下范围是本项目提出的初始值，不来自 TAC 全量实现：

| 参数 | 初始范围 |
|---|---|
| clip duration | 10–30 s |
| active sources | 1–8 |
| max concurrency | 1–5 |
| foreground/background relative SNR | -10–20 dB |
| T60 | 0.1–1.2 s |
| echo delay | 80–500 ms |
| echo attenuation | -18–-3 dB |
| event repeat | 1–5 |
| merge threshold | U(0.1, 1.0) s |
| output resolution | {0.1, 0.5, 1.0} s；0.01 只做消融 |
| style | keyword/brief/detailed |

R1 前 20% steps 以 1–3 sources 为主，随后增加 overlap/退化；否则模型在尚未学会格式前同时面对严重声学干扰，parser failure 会主导训练。

## 4. Template 设计

### Paper-spec templates

1. isolated sfx/ambience；
2. continuous music + non-overlapping speech；
3. continuous ambience + intermittent sfx；
4. music + sfx；
5. one speech stream + background + sfx；
6. repeated same event。

### Complex extensions

1. two/three overlapping speakers；
2. speech over song with lyrics；
3. lyrics + accompaniment + foreground sfx；
4. speech/lyrics type confusion pairs；
5. source removal negative；
6. echo of one speaker，不增加 speaker count；
7. short transient masked by continuous foreground；
8. visually present but silent，供后续 AV teacher 使用。

每个 batch 记录 template distribution；论文中报告每类比例。随机选择声音后相加不等于 scene-template baseline。

## 5. Prompt 与 target

### 5.1 Canonical prompt

```text
Describe every audible event in the audio.
Return speech, sung lyrics, music, and sound effects as separate typed events.
Events may overlap and may contain multiple time spans.
[style={style}, merge={merge}s, activity={activity}, resolution={resolution}s]
Do not infer events that are not acoustically supported.
```

`B2-paper-spec` 可不要求 lyrics；`B2-complex/B3` 加入 lyrics。训练中 10% prompt 做等义改写，90% 保持 canonical，以避免 prompt robustness 和任务学习混为一谈。

### 5.2 Target order

1. onset 升序；
2. onset 相同按 `speech, lys, music, sfx` 的固定类型顺序；
3. 同类型按 source ID；
4. 一个 event 的多个 spans 放在同一 tag；
5. 没有事件输出 `<empty/>`，禁止编造背景噪声填充。

稳定 order 只服务自回归基线。SceneLedger set loss 不依赖该顺序，但 serializer 使用同一规则，保证输出可比较。

## 6. 训练配置

### 6.1 主 run

```yaml
model: OpenMOSS-Team/MOSS-Audio-4B-Instruct
precision: bf16
freeze_audio_encoder: true
freeze_base_llm: true
lora:
  rank: 128
  alpha: 256
  dropout: 0.05
  audio_encoder: false
optimizer:
  name: AdamW
  peak_lr: 5.0e-5
  schedule: cosine
  warmup_steps: 1000
train:
  steps: 5000
  global_effective_batch: 32
  max_audio_seconds: 30
loss:
  text_weight: 1.0
  type_weight: 2.0
  timestamp_weight: 5.0
```

若显存不足，用 gradient accumulation 保持 effective batch；不要先改变 global batch 再比较。MOSS 官方 LoRA 默认 alpha=2，与 TAC 的 alpha=256 不同，因此 B2 主 run 明确覆盖默认值，同时做 `rank64/alpha128` 的资源友好版本。

### 6.2 必做 runs

| ID | 改动 | 要回答的问题 |
|---|---|---|
| B0 | zero-shot MOSS | backbone 原始能力 |
| B1 | static prompt + ordinary CE | 统一格式 SFT 是否已足够 |
| B2 | dynamic prompt + weighted CE + templates | TAC recipe 总收益 |
| B2-no-template | 随机混音 | scene templates 是否必要 |
| B2-time1 | timestamp weight 1 | 时间加权是否必要 |
| B2-time10 | timestamp weight 10 | 过强时间 loss 是否伤语义 |
| B2-no-TACOS | 不用 TACOS train | 真实 strong captions 的贡献 |
| B2-10k | 10k steps | 是否出现 synthetic overfitting |
| B3 | + overlapping speech/lyrics joint targets | 任务扩展，不改变结构时的上限 |

至少为 B1、B2、B3 跑 3 个 seeds。资源有限时，消融先单 seed 筛选，主结论再补 3 seeds。

## 7. 指标实现

### 7.1 解析有效性

- valid-tag rate；
- timestamp token 是否成对；
- start ≤ end、范围在 clip 内；
- canonical JSON/XML round-trip 是否完全一致；
- duplicated event 和非法 source ID 比例。

主 run 要求 validation parser success ≥99.5%。否则性能变化可能只是格式错误。

### 7.2 TAC-compatible 指标

- `SegF1@100ms`；
- `EvtF1@onset±1.0s`；
- FLAM-based `Hal%/confidence/specificity`；
- semantic matching 的 judge prompt、模型版本、temperature 和缓存全部冻结。

同时新增更严格但不替代原指标的：

- EvtF1 onset collar 0.1/0.25/0.5/1.0 s；
- offset collar 与 joint onset-offset F1；
- boundary median/p90 absolute error；
- multi-span IoU；
- type confusion matrix；
- negative/silence false-event rate。

### 7.3 speech/music/lyrics

- speech：WER/CER、cpWER/tcpWER、DER/JER，重叠区单独报告；
- lyrics：line CER/WER、lyric presence F1、line boundary MAE；
- music：instrument/genre/mood/structure 的分项 factual score；
- 不可听样本：abstention accuracy 和 selective risk，而不是强制 WER。

## 8. 复现验收

由于 backbone/data 不同，不把“达到 TAC 报告的 EvtF1=.50”设为唯一成功条件。按以下顺序验收：

1. 20 个手工 toy cases 的 parser、SegF1、EvtF1 与匹配结果逐例正确；
2. R0 上达到近乎完美的格式与边界恢复，证明实现没有系统性 bug；
3. B2 相对 B1 在 TACOS test 与人工 pilot 的 EvtF1/SegF1/Hal 中至少两项改善；
4. dynamic prompt 能按 resolution/style 控制输出，而非忽略 prompt；
5. 10k 相对 5k 出现的趋势被记录，不挑最佳 checkpoint 后假称固定 5k；
6. 三个 seeds 报 mean±std；
7. Qwen2-Audio 小规模 sanity run 与 MOSS-B2 的主要趋势一致。

若第 3 项失败，不推进 SceneLedger 大模型训练。优先检查：TACOS 切分、semantic matcher、timestamp token 是否真为 atomic、LoRA 是否包含新增 embedding/lm head、mixer activity 是否与听感一致。

## 9. 防止数据泄漏

- 按原始媒体 ID、音频指纹、上传者、歌曲/演讲者和 source stem 分组；
- 同一 dry source 的不同混音不能跨 train/test；
- TACOS/Freesound source 若已进入训练 mix，不能再以另一裁剪出现在 test；
- web 视频以音频 fingerprint + perceptual video hash 去重；
- teacher 生成文本不得读取隐藏 benchmark reference；
- 所有 split 生成脚本固定 seed 并输出 group-level audit report。

## 10. 从 B2 迁移到 SceneLedger

B2 完成后可复用：

- 数据 manifests、mixer、atomic token parser、评测代码；
- MOSS LoRA 初始化，作为 text decoder 起点；
- B2 输出作为 hard-negative mining 候选；
- B2 高置信/低置信分桶，用于构造 CARC curriculum。

不应直接复用：

- B2 生成的时间作为 event-slot 真值；真实无标注数据上它只是 teacher；
- B2 的自回归顺序作为 slot ID；
- FLAM threshold 0.25 作为所有 type 的统一人工真值。

最终论文表格保持 B2、B3、S1、S2、S3 的连续路线，才能把收益归因到 set prediction、evidence conditioning 和 CARC。

