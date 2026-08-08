# 开源底座选择与 SceneLedger 实现蓝图

> 决策冻结日期：2026-08-08。本文把“可直接运行的工程底座”“论文必须比较的科学基线”和“最终方法”分开，避免把复用开源模型误写成方法贡献。

## 1. 最终底座决策

**主工程底座选用 `OpenMOSS/MOSS-Audio-4B-Instruct`，在其上先实现 TAC-style 基线，再实现 SceneLedger。**

原因不是 MOSS-Audio 已经解决了本任务，而是它最适合承载本任务：

- [官方仓库](https://github.com/OpenMOSS/MOSS-Audio)公开模型权重、推理、LoRA/全参微调代码，并声明模型采用 Apache-2.0；
- 4B 版本比 7B/8B backbone 更适合大量结构和数据消融；
- encoder 输出 12.5 Hz 表征，80 ms 帧间隔足以支撑 100 ms 输出网格；
- 预训练已经同时覆盖 speech、general sound、music、singing 与 timestamp ASR；
- `get_audio_features()` 暴露最后层和 DeepStack 中间层，便于接 event slot head，而不必重写音频前端；
- 官方 `finetune/finetune.py` 已支持 LoRA、audio-encoder LoRA、DeepSpeed 和标准 JSONL conversation 数据。

这里的“底座”只表示代码和初始化权重。论文中的关键结果仍必须来自统一任务数据、event-set 建模、局部证据约束与 CARC，而不是声称 MOSS-Audio 本身是新方法。

## 2. 开源候选审计

| 工作 | 代码/权重状态 | 许可证与工程约束 | 在本项目中的角色 |
|---|---|---|---|
| [MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) | 完整推理、4B/8B 权重、LoRA/全参训练脚本 | 官方声明 Apache-2.0；Python 3.12；16 kHz 输入；官方默认 variable-length collator 要求单卡 batch size 1 | **主底座** |
| [TAC](https://sonalkum.github.io/tacmodel/) | 官方页面截至冻结日只有论文、结果与演示，未发现训练代码或 checkpoint 入口 | 使用未公开的 licensed single-source corpus；因此无法做严格 bitwise reproduction | 按论文规格重实现，作为最重要科学基线 |
| [SpotSound](https://github.com/LoieSun/SpotSound) | 已公开 train/inference、checkpoint，基于 Audio Flamingo 3 | 适合 query grounding；AF3 checkpoint 是 NVIDIA OneWay Noncommercial；其 HF benchmark 页面当前只有约 4 KB metadata，音频数据未完整出现 | 复用 negative-query 训练思想和 grounding baseline，不作主底座 |
| [Audio Flamingo 3](https://github.com/NVIDIA/audio-flamingo) | 训练/推理代码和权重公开 | 代码 MIT，但 checkpoint 为非商业研究许可证；7B 成本高于 MOSS-4B | 外部强基线；也是 SpotSound 运行依赖 |
| [TimeAudio](https://huggingface.co/lysanderism/TimeAudio) | HF 有约 907 MB 增量 checkpoint 和 README | 截至冻结日未发现与 checkpoint 配套的完整训练仓库；基于 SALMONN | 能运行时作为 temporal baseline，不承担主开发 |
| [SAM Audio](https://github.com/facebookresearch/sam-audio) | 公开推理代码和需申请访问的 checkpoint | 自定义 SAM License；显存和推理开销较大 | 只作为高价值 pseudo-stem teacher，不成为训练/推理依赖 |
| [WhisperX](https://github.com/m-bain/whisperX) | 完整 ASR、word alignment 与 diarization pipeline | BSD-2-Clause；diarization 还涉及 pyannote 模型访问与条款 | 产生 speech pseudo labels 和评测参考 |
| [Demucs](https://github.com/facebookresearch/demucs) | 完整 music source separation 代码与模型 | MIT；仓库已归档但实现成熟 | 免费、可批处理的 vocals/music stem teacher |

选择 MOSS 而不是 SpotSound/AF3 的关键原因是：SpotSound 只接受一个 query 并预测其是否存在和时间，不直接生成全量 event ledger；AF3 checkpoint 的非商业条款会增加后续数据和模型发布的不确定性。二者仍应进入论文对比。

## 3. 三层系统，而不是一次写完最终模型

### 3.1 B0/B1/B2：可审计基线

- `B0-moss-zero-shot`：原始 MOSS-Audio-4B-Instruct，只用统一 caption prompt；测当前模型能力和解析失败率。
- `B1-moss-static-sft`：固定 `brief + 0.1 s` 输出格式，用普通 token CE 做 LoRA；验证数据本身是否有效。
- `B2-moss-tac`：加入 TAC 的 Dynamic Acoustic Mixer、多任务 style/merge/activity/resolution prompt、atomic timestamp token 和时间加权 CE。这是后续 SceneLedger 的主基线。
- `B3-moss-tac-joint`：在 B2 上加入 `<lys>`、多说话人 speech 和不调用外部 Whisper 的联合文本目标，用于区分“任务扩展收益”和“SceneLedger 结构收益”。

### 3.2 S1/S2/S3：SceneLedger 主模型

- `S1-ledger`：共享 MOSS encoder，新增 permutation-invariant event slot decoder、活动 mask 和边界头；文本暂用 B3 结果或简单 event decoder。
- `S2-evidence`：加入 slot-local evidence-conditioned text decoder、source embedding、null/eventness 与校准头。
- `S3-carc`：在 S2 上加入 source removal/addition/shift/nuisance 的 CARC 训练，是完整论文模型。

每次升级只改变一个主要因素。禁止把 backbone、数据规模、标注质量和模型结构同时改变后只报告最终数字。

## 4. 任务作用域与规范输出

### 4.1 第一版作用域

- 训练片段：10–30 s，单声道 16 kHz 进入 MOSS；分离和音乐 teacher 可保留 44.1/48 kHz 原始副本；
- 最多 24 个 event slots；超过上限的场景只保留可听度最高的 24 个并记录 truncation，不静默丢弃；
- event onset/offset 使用 0.1 s 输出网格；内部表示可以是 80 ms 或连续时间；
- `<speech>`/`<lys>` 第一阶段提供 utterance/line 级时间，word-level 时间作为附加头和附加指标；
- 长音频先使用 30 s window、5 s overlap、source embedding 跨窗合并，论文主表仍在不需要跨窗拼接的片段上比较。

### 4.2 标签语义

- `<speech>`：有语言内容的说话、念白、对话；
- `<lys>`：被唱出的可辨识歌词；低置信时输出 `<music vocal="present">含不可辨认演唱</music>`，不能猜词；
- `<music>`：伴奏、器乐、曲风、结构、节奏、乐器、动态以及非词汇性人声；
- `<sfx>`：环境声、物体声、动物声、机械声和具有语义的声学事件；
- noise/reverb/codec 默认是 recording condition，不作为独立 event；明显且有语义的回声可作为属性 `echo="salient"`，不能被误判为第二说话人。

canonical JSON 是训练与评价真值，XML-like 文本只是确定性序列化。例如：

```xml
<music id="M1" t="0.0-12.8">轻快的电子伴奏，鼓点逐渐增强。</music>
<speech source="S1" t="0.7-2.9">我们现在开始。</speech>
<lys source="V1" t="3.2-6.1">take me home tonight</lys>
<sfx source="E1" t="4.6-4.9">近处一次玻璃破碎声。</sfx>
```

## 5. B2：在 MOSS 上实现 TAC-style 基线

### 5.1 Atomic timestamp vocabulary

对 30 s clip 增加 301 个 decisecond token：`<|t_000|>` 到 `<|t_300|>`，分别表示 0.0–30.0 s。再加入四个类型 token 和结构 token。训练 target 内部写为：

```text
<sfx> <|t_046|> <|t_049|> 近处一次玻璃破碎声 </sfx>
```

展示或导出时再转换为 `t="4.6-4.9"`。不要直接训练浮点字符串，否则 `4.6` 可能被分成多个普通 token，也无法稳定施加 timestamp loss。

执行要点：

1. `tokenizer.add_special_tokens(...)`；
2. `model.resize_token_embeddings(len(tokenizer))`；
3. LoRA 之外必须让新增 embedding rows 和 `lm_head` 可训练；使用 PEFT `modules_to_save` 或显式梯度 mask；
4. 保存 tokenizer、special-token map 和 checkpoint；任一缺失都会导致推理 token ID 不一致。

### 5.2 Weighted CE

MOSS 官方 `forward()` 当前使用统一 `CrossEntropyLoss`。B2 新增逐 token loss：

$$
L_{tok}=\frac{\sum_j m_j w(y_j)\operatorname{CE}_j}{\sum_j m_j w(y_j)},
$$

其中 padding/user/audio token 的 $m_j=0$；普通文本 `w=1`、类型 token `w=2`、timestamp token 初始 `w=5`。必须按总权重归一化，否则增加事件数会隐式改变学习率。主复现使用 `timestamp_weight=5`，并做 1/5/10 消融。

实现上优先新增 `WeightedTokenTrainer.compute_loss()`，不要直接改 vendor 源文件；trainer 调用模型取得 logits，再以 `reduction="none"` 计算 loss。这样同一 MOSS checkout 可同时跑 B1 和 B2。

### 5.3 Dynamic mixer

每条样本保存 waveform、每个 dry/wet stem、RMS activity、scene seed 和规范 ledger。建议模板至少包括：

1. isolated/near-isolated sanity；
2. speech over music；
3. two-speaker overlap；
4. music + lyrics + sfx；
5. foreground transient over ambience；
6. repeated same event；
7. harsh acoustics：RIR/echo/noise/codec；
8. negative/silence/very-low-audibility。

TAC 论文中的 speech streams 不互相重叠，本复现分两套报告：`B2-paper-spec` 保持该限制，`B2-complex` 允许重叠 speech。这样可判断多说话人差距是数据定义还是模型能力造成的。

## 6. SceneLedger-MOSS 模型结构

### 6.1 Wrapper，而不是 fork 整个 MOSS

新增 `SceneLedgerMoss(nn.Module)` 包装官方 `MossAudioModel`：

```text
MossAudioModel.get_audio_features()
  ├─ last_hidden_state [B, T80, D]
  └─ deepstack_hidden_states [L, B, T80, D_l]
             │
      TemporalFeatureFusion
             │  interpolate/project to [B, T100, 768]
      EventSlotDecoder (K=24)
       ├─ type/null
       ├─ 100-ms activity mask
       ├─ onset/offset distributions + uncertainty
       ├─ source embedding
       └─ audibility/confidence
             │
      EvidencePrefixProjector
             │
      shared MOSS/Qwen3 text decoder + type adapters
             │
      deterministic serializer
```

MOSS 内部是 12.5 Hz。`TemporalFeatureFusion` 可线性插值到 10 Hz，或保留 12.5 Hz 并预测连续 offset 后量化到 0.1 s。主实验需要比较这两个选择，不能把 80 ms encoder frame 自动解释为 100 ms 准确度。

### 6.2 推荐初始超参数

| 模块 | 初始设置 | 必做消融 |
|---|---:|---|
| slots | 24 | 12 / 24 / 36 |
| slot decoder | 6 层、hidden 768、8 heads | 3 vs 6 层 |
| temporal grid | 10 Hz | 原生 12.5 Hz + continuous offset |
| source embedding | 256 dim, L2 normalized | 无 source loss |
| evidence prefix | 每个 event 4 tokens | 1 / 4 / 8 |
| global context | 2 tokens，经 scalar gate | 无 global / unrestricted global |
| text adapters | shared LLM + 4 类 LoRA/adapters | 完全共享 |

这些是工程起点，不是论文事实；pilot 后由 dev set 冻结，hidden test 不参与选择。

### 6.3 Event matching 与 loss

对预测 slot $k$ 与真值 event $i$ 定义：

$$
C_{ki}=\lambda_z C_{type}+\lambda_m C_{mask}+\lambda_b C_{boundary}+\lambda_q C_{source}.
$$

Hungarian matching 后计算：

- type/null：weighted CE 或 focal loss；
- activity：frame BCE + Dice，解决短事件占比低；
- boundary：对标注允许区间的 soft NLL，加连续偏移 L1；
- source：同源拉近、异源拉远的 supervised contrastive loss；
- audibility/confidence：Brier loss；
- 未匹配 slots：只训练 null/eventness，不生成文本。

文本生成不直接读取整个音频。对 slot mask $a_{kt}$：

$$
z_k=\frac{\sum_t a_{kt}h_t}{\epsilon+\sum_t a_{kt}},
$$

再把 $z_k$ 投影为 4 个 evidence prefix tokens。训练时已匹配 events 可批量 teacher-forcing；推理时所有有效 slots 共享一次 encoder 结果并批量解码，最后合成一个 caption。系统对用户仍是单次 audio-to-caption 调用，但内部不是把所有重叠事件塞进一条不可解释的自回归时间序列。

## 7. 训练数据接口

原始 ledger JSONL 建议：

```json
{
  "sample_id": "mix_000001",
  "audio_path": "audio/mix_000001.wav",
  "duration": 12.8,
  "events": [
    {
      "event_id": "E1",
      "type": "sfx",
      "source_id": "glass_001",
      "spans": [[4.6, 4.9]],
      "text": "nearby glass breaking",
      "audibility": 0.91,
      "boundary_uncertainty": [[4.55, 4.65, 4.85, 4.95]]
    }
  ],
  "conditions": {"snr_db": 3.0, "t60": 0.7, "codec": "opus"},
  "provenance": {"kind": "synthetic", "scene_seed": 1947}
}
```

数据 converter 生成 MOSS 官方 conversation JSONL。不要把 pseudo label 与人工/精确 stem label 混成同一布尔字段；`provenance.kind`、teacher、置信度、许可证和可再分发状态必须随样本保留。

## 8. 无标注视频 teacher 栈

推荐使用“廉价 teacher 全量跑，高成本 teacher 只跑难例”的级联：

1. `ffmpeg` 解复用，保存原始采样率副本和 16 kHz mono；
2. WhisperX：speech/word time/初始 speaker attribution；
3. Demucs：vocals 与 accompaniment，提供 `<lys>` 候选和 music-vocal remix；
4. MOSS-Audio：全局/分段语义候选，不直接视为真值；
5. FLAM/open-vocabulary SED：验证事件文本是否在指定局部有证据；
6. SAM Audio：只对高价值、teacher 冲突或需要 source removal 的样本做 prompt separation；
7. 视频 VLM/AV sync：只提供“可能是什么/是否 on-screen”的辅助证据，不能单独创建正事件。

所有阈值都在 200 条人工 pilot 上校准。不要先拍脑袋固定 `FLAM > 0.25` 就处理百万视频；不同 type、可听度和 teacher 的分数分布不同。

## 9. 代码目录建议

```text
complex-audio-caption/
├─ configs/
│  └─ experiment_matrix.yaml
├─ sceneledger/
│  ├─ data/          # ledger dataset, MOSS converter, samplers
│  ├─ mixer/         # scene templates, RIR/echo/noise/codec
│  ├─ models/        # MOSS wrapper, slotter, evidence decoder
│  ├─ losses/        # weighted CE, set loss, CARC
│  ├─ decoding/      # constrained parser/serializer
│  └─ evaluation/    # ST-F1, SegF1, EvtF1, WER/DER, calibration
├─ scripts/
│  ├─ prepare_*.py
│  ├─ train_moss_tac.py
│  ├─ train_sceneledger.py
│  └─ evaluate.py
└─ tests/
   ├─ test_timestamp_tokens.py
   ├─ test_ledger_roundtrip.py
   ├─ test_mixer_determinism.py
   └─ test_metric_toy_cases.py
```

MOSS、SpotSound、SAM Audio 等上游仓库不复制进本仓库。用锁定 commit 的安装说明或 git submodule，并保留许可证/NOTICE。初期优先锁定 commit SHA，避免上游更新导致实验不可复现。

## 10. 资源分层

- **smoke tier**：1×48/80 GB，10 s 音频、LoRA rank 32、小数据，只验证数据/格式/loss；
- **baseline tier**：建议 4×80 GB，30 s、LoRA rank 64/128，完成 B0–B3；
- **paper tier**：建议 8×80 GB，S1–S3、全消融和多 seed。

TAC 原论文使用 8×A100-80GB、global batch 32、LoRA rank 128/alpha 256、AdamW peak LR `5e-5`、1000 warmup 和 5k steps。B2 主复现先沿用这些可观察参数，但 MOSS backbone 与数据不同，结果只能称为 **paper-spec reimplementation**，不能声称 exact reproduction。

## 11. 必须提前锁定的失败标准

1. B2 相对 B1 若在 TACOS/人工 pilot 的 EvtF1、SegF1、Hal 三项中没有至少两项改善，不进入 event slots；先修 mixer/metric。
2. S1 相对 B3 若仅提高合成集、不提高 200 条真实 pilot，不扩充 slots 和模型规模。
3. S2 若降低 hallucination 却使 speech cpWER 或 lyrics CER 恶化超过预设容忍线，检查 evidence mask 过窄，而不是直接删掉 speech/lyrics。
4. S3 若 CARC 只在 pseudo-stem 数据有效、真实 stem/人工集无效，停止扩大互联网数据，先修 leakage 与 audibility gating。
5. 三个 seed 的结论方向不一致时，不写“显著提升”。

