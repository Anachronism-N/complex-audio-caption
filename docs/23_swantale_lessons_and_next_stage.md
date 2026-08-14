# SwanTale 对 SceneLedger 的启示与下一阶段实验

> 调研日期：2026-08-12。SwanTale 当前为 arXiv v2 预印本；本文区分论文作者报告、可公开复用组件和本项目自己的推论。

## 0. 决策摘要

下一步实验仍然是 [`B3-complex-v2` 数据锚点生成与验收](22_next_experiment_b3_complex_v2_data_anchor.md)，**现有代码足够，不新增模型代码**。原因是最新的 3k/5k 训练结果仍来自泄漏且质量门禁失败的旧 `b3_5k` 数据；在新数据合同通过前继续训练无法回答模型问题。

SwanTale 不适合作为我们的代码 base：它研究的是 **caption/reference → audio** 的统一生成，而我们研究 **audio → timestamped unified caption**；截至本次调研，论文和项目页也没有提供可直接复用的 SwanTale 训练代码、checkpoint 或 SwanData-Caption 数据。其核心训练还依赖内部数据、Seed-ASR 2.0、Seed2.0 Lite、SwanAligner 和内部 SwanVerifier。

但 SwanTale 对我们有三项立即有效的启示：

1. 把输出语义分成 `Environment / Speakers / chronological Content`，补足当前 Ledger 对声学环境、稳定说话人属性和副语言变化的覆盖；
2. 分离、diarization、ASR、VLM 都只能提供**可审计证据**，不能直接成为 ground truth；低置信度必须 abstain；
3. 数据课程应从可靠的单域锚点逐步扩展到真实全混合场景，最后才做 RL，而不是一开始同时优化大量不可靠 reward。

因此本项目的主张需要收敛为：

> **Evidence-Grounded Inverse Scene Captioning**：从真实复杂混音中，一次性恢复场景环境、稳定声源/说话人、按时间排序的 speech/music/lyrics/SFX 事件及其局部声学证据，并输出可读的统一全局 caption。

这比“统一处理 speech/music/sound”更准确，也与 SwanTale 的生成任务形成互补而非重复。

## 1. SwanTale 做了什么

论文：[SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks](https://arxiv.org/html/2608.02023)

### 1.1 任务

SwanTale 统一两种生成路径：

- **instruct generation**：输入完整自然语言 caption，生成包含多说话人语音、环境、局部音效，偶尔包含歌声与音乐的波形；
- **zero-shot generation**：输入参考音频和 content caption，保留参考说话人的声音特征并生成新内容。

它的输入 caption 可以非常接近我们期望的最终全局描述，但方向相反：SwanTale 把 caption 变成音频，我们把音频变成 caption。

### 1.2 SwanData-Caption 数据流程

SwanTale 的数据流程分四块：

1. **Coverage design**：真实 media-style 数据为主体，另用目标明确的合成数据补长尾，而不是让大规模规则混音替代真实分布；
2. **Speech preprocessing**：先把 vocal 和 residual background 分开，在 vocal 上做 diarization、ASR 和 alignment，同时始终保留原始 mixture 供场景 caption；
3. **Caption annotation**：audio + 去标点 ASR transcript + 约束 prompt 共同输入标注器，生成严格结构化 caption；
4. **Data refinement**：waveform quality filtering、结构检查、属性验证和人工审计共同筛选。

值得注意的是，它明确承认 diarization 不够可靠，因此 diarization 只用于粗分段；最终 speaker discrimination 由后续 caption annotation 与审计处理。这与我们“显式分轨可以是 teacher，但不能把伪分轨当真值”的原则一致。

### 1.3 三层 caption 定义

SwanData-Caption 使用三个字段：

| 字段 | 内容 | 时间尺度 |
|---|---|---|
| `Environment` | 地点/空间、声场、录音条件、混响、持续背景音乐与环境床 | clip/scene 级 |
| `Speakers` | 实际发声者的稳定属性：感知年龄、性别、音色、口音、习惯语速、角色/人格等 | track/entity 级 |
| `Content` | 按时间顺序的说话内容、speaker turns、局部情绪/音量/语速/停顿变化和局部音效 | event/utterance 级 |

最重要的标注原则是：

- 稳定说话人特征放在 `Speakers`；
- 瞬时情绪、强调、停顿、打断和局部音效放在 `Content`；
- persona/role 只有在语音内容或听觉表现提供证据时才允许出现；
- ordinary data 不强行套用 persona style matrix。

这正好回答了我们此前的疑问：当前 `<speech>/<music>/<lys>/<sfx>` 事件列表不足以承载全局环境和稳定说话人特征，但不需要放弃事件账本；应在事件账本上增加 scene/entity 两层，再由三层结构重写出统一全局 caption。

### 1.4 SwanVerifier 的原则

SwanVerifier 是 WavLM backbone 加属性特定 attention-pooling heads 的轻量模型，主要核验 age group 和 perceived gender；emotion、pitch 和 speaking rate 只作为辅助证据。它的关键不是模型结构，而是**选择性验证**：

- 只在单一且可分离的 speaker segment 上做硬核验；
- 只有预测置信度超过属性阈值时才比较；
- confident mismatch 只触发 repair/re-caption/remove；
- 低置信、重叠未解决、caption 未声明该属性时一律不自动判断；
- 不自动补写缺失属性；复杂 persona 与局部情绪留给人工审计。

这比“让一个 LLM/VLM 再看一遍并自动纠正”更可靠。验证器的职责是发现矛盾和拒答，不是生成更丰富但可能没有证据的描述。

### 1.5 模型与训练

SwanTale 还包括：

- 48 kHz、25 Hz continuous latent 的 SwanVAE；
- non-causal flow-matching DiT；
- 分离 caption control、spoken text alignment 和 speaker-turn embeddings；
- label embeddings 区分 environment、speech content 和 local audio effect；
- sample-level task router 与 frame-level audio router 的 Unified MoE；
- reward-conditioned quality control；
- Engram n-gram memory 加强重复、稳定的 caption phrase；
- 从单说话人到多说话人、clean caption、full mixture、high-quality SFT 的 curriculum；
- 最后用 GRPO 优化 pronunciation、pause、boundary energy、waveform quality 和 speaker control。

这些模块服务的是生成模型，不能原样移植到 MOSS captioner。对我们真正有用的是“条件/输出角色分离”“课程顺序”和“只有可验证 reward 才进入 RL”。

## 2. 与 SceneLedger 的相同点和差异

| 维度 | SwanTale | SceneLedger 目标 | 结论 |
|---|---|---|---|
| 方向 | caption/reference → waveform | waveform → caption/ledger | 互补任务 |
| 模态 | speech、audio、偶发 singing/music | speech、music、lyrics、SFX 全覆盖 | 有重叠，但我们的 music/lyrics 理解更核心 |
| 多说话人 | structured speaker tags/turn conditioning | speaker tracks、speaker-attributed events | 可借 speaker inventory 与 turn consistency |
| 环境 | `Environment` 全局字段 | 当前主要只有 `conditions` 数值字段 | 需要补自然语言 scene layer |
| 副语言 | stable speaker profile + local delivery | 当前主要塞在自由文本/attributes | 需要稳定/瞬时属性分层 |
| 时间 | 内容按顺序，生成目标不以 0.1 s 标注为核心 | 每个事件显式 0.1 s span | 我们的关键差异 |
| 证据 | 分离、ASR、alignment、verifier、人工审计 | local evidence、CARC、AV weak evidence | 可借鉴并进一步形式化 |
| 幻觉 | instruction-following generation | 不存在事件/错误属性/错误时间 | 我们需要更严格的 existence 与 evidence 评价 |
| 数据 | 内部超大规模、真实 media 为主 | 可用互联网无标注数据 + 开放语料 | 方法可借，资源不可复现 |

SwanTale 使以下说法不再适合作为论文主创新：

- “第一个统一 speech、sound、music 的系统”；
- “第一个描述环境和说话人风格的细粒度 caption”；
- “第一个在复杂媒体音频中加入多说话人与局部音效”。

仍然成立且更清晰的差异是：

- 同一混音的一次性**逆向解析**；
- 事件存在性、类型、语义、speaker attribution 和 0.1 s 时间同时输出；
- 每个局部描述绑定可审计 acoustic evidence；
- removal/addition/time-shift/nuisance intervention 下事件集合满足反事实约束；
- 在真实复杂音频上分别评价 hallucination、omission、speaker error、semantic error 和 boundary error。

## 3. 建议的双视图输出

### 3.1 内部 canonical representation

当前 `Ledger v0.2` 不应立即修改。完成数据锚点后，再设计 `v0.3`，把现有两层扩成三层：

1. **Scene layer**：全局环境、空间/录音条件、持续 background bed；
2. **Entity/Track layer**：speaker/singer/music bed 等持续身份及稳定属性；
3. **Event layer**：带 span 的 speech/lyrics/music change/SFX 和局部副语言属性。

建议概念结构如下；它是设计草案，不是当前可直接训练的 schema：

```json
{
  "scene": {
    "environment_text": "一间混响明显的室内空间，背景有持续的轻音乐",
    "attributes": {"reverb": "strong", "noise": "low"},
    "confidence": 0.82,
    "evidence": {"method": "audio_teacher_ensemble", "audio_support": 0.86}
  },
  "entities": [
    {
      "id": "S1",
      "kind": "speaker",
      "stable_description": "一名声音低沉、语速较快的成年说话者",
      "attributes": {"pitch": "low", "habitual_rate": "fast"},
      "confidence": 0.79
    }
  ],
  "events": [
    {
      "type": "speech",
      "track_id": "S1",
      "spans": [{"start_sec": 0.7, "end_sec": 2.9}],
      "text": "我们现在开始",
      "attributes": {"emotion": "excited", "delivery": "emphatic"},
      "confidence": 0.91
    }
  ]
}
```

### 3.2 用户可读统一 caption

最终全局 caption 是 canonical representation 的**确定性/受约束重写视图**，不是另一个独立 ground truth。示例：

```xml
<scene>一间混响明显的室内空间，背景有持续的轻快电子音乐。</scene>
<speaker id="S1">一名声音低沉、语速较快的成年说话者。</speaker>
<music id="M1" t="0.0-12.8">电子伴奏持续播放，鼓点逐渐增强。</music>
<speech speaker="S1" t="0.7-2.9" delivery="兴奋、强调">我们现在开始。</speech>
<lys singer="V1" t="3.2-6.1">take me home tonight</lys>
<sfx id="E1" t="4.6-4.9">近处传来一次玻璃破碎声。</sfx>
```

这样既保留用户最初要求的 `<music>/<lys>/<sfx>/<speech>` 和时间戳，也不会丢失 scene、speaker 和 paralinguistic 信息。重写器只能使用 Ledger 中已有字段，禁止增加未被证据支持的新事实。

### 3.3 为什么不立即改 schema 和代码

当前最紧迫的问题是旧数据失败和评测泄漏，不是字段数量不够。现在修改 schema 会同时改变 renderer、parser、formatter、训练 target 和 evaluator，使 `B3-complex-v2` 锚点无法回答单一问题。

正确顺序是：

1. 先用 v0.2 完成数据工程锚点；
2. 用 200 条真实样本验证 scene/speaker/local attribute 是否能可靠标注；
3. 只有人工审计证明三层字段可用，才冻结 v0.3 schema 并写 migration；
4. 用同一批样本比较“event-only”与“三层双视图”训练。

## 4. 从互联网音视频构造标注的具体流程

### 4.1 Stage W0：合规与 group identity

每条原始媒体先写最小 manifest：

```json
{
  "media_id": "platform:video_id",
  "audio_path": "/absolute/path/audio.wav",
  "video_path": "/absolute/path/video.mp4",
  "platform": "bilibili|instagram|tiktok|other",
  "uploader_id": "stable uploader hash",
  "start_sec": 12.0,
  "end_sec": 30.0,
  "language_hint": "zh",
  "license_status": "research_only|redistributable|unknown",
  "source_group": "recording/song/speaker/uploader group"
}
```

先按原始 video、audio fingerprint、uploader、speaker、song/performer 建 group，再划 train/val/test。不能先切片、分离或生成 pseudo-label 后再随机拆分。

### 4.2 Stage W1：保留 mixture 的辅助分离

目标不是取得“真实 stems”，而是产生两条辅助证据：

- vocal stream：提高 diarization、ASR、speaker attribute 的可靠性；
- residual stream：提高 music、ambience 和 local SFX 的可见度。

原始 mixture 永远是 caption 的主输入和最终推理输入。分离结果必须保存：

- separator name/version/checkpoint hash；
- vocal/residual waveform URI；
- reconstruction error 或 mixture–sum residual；
- separation artifact flag；
- 每个局部证据的置信度。

可用组件包括 [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) 或现有 Demucs 路径。SwanTale 的 UVR 使用方式可以借鉴，但分离失败绝不能删除原始 mixture 中实际存在的事件。

### 4.3 Stage W2：speech anchors

在 vocal stream 上执行：

1. VAD；
2. diarization；
3. ASR；
4. forced alignment；
5. speaker-attributed transcript；
6. emotion/pitch/rate 等弱属性。

可公开替代方案：

- [3D-Speaker](https://github.com/modelscope/3D-Speaker)：VAD、CAM++ embedding、clustering 和可选 overlap detection；
- [SenseVoice](https://github.com/QwenAudio/SenseVoice)：中/粤/英/日/韩 ASR、emotion 与有限 audio-event tags；
- Whisper/WhisperX 或其他可复现 aligner：补充词级时间和跨语言覆盖；
- WavLM/ECAPA speaker encoder：speaker consistency，不直接生成 persona。

强制规则：

- diarization 只给匿名 `S1/S2/...`，不推断真实身份；
- speaker IDs 按首个可靠 utterance 排序且连续；
- ASR 文本在 LLM 重写前后去标点归一化后必须一致；
- overlap 中无法确定归属时标 `unresolved`，不强行分给某个 speaker；
- emotion 只允许出现在局部 event attributes；稳定风格需要跨多个 utterance 支持。

### 4.4 Stage W3：music、lyrics 和 SFX anchors

- 在 mixture 和 residual 上分别运行 audio event/music teacher；
- lyrics 候选必须同时满足 vocal activity、歌词 ASR/aligner 与 music/singing classifier；
- local SFX 必须有局部 audio support，不能只由 VLM 看到物体而写入；
- music style、instrument、tempo 等细节按属性分别记录置信度，不能用一个总体 confidence 覆盖；
- ambience/persistent background 放 scene layer；短暂或发生变化的声音放 event layer。

若视觉可用，VLM 只提供候选和场景上下文：

- 视觉+音频一致可提高 review priority 或 AV support；
- 视觉没有看到声源不能否定 off-screen sound；
- 视觉看到狗、车、乐器不能证明它正在发声；
- 所有正事件仍需要局部 audio evidence。

### 4.5 Stage W4：受约束 caption annotation

标注器输入：

- 原始 mixture；
- vocal/residual 辅助流；
- 去标点 speaker-attributed ASR；
- diarization/forced-alignment spans；
- music/SFX/lyrics teacher 候选及置信度；
- 可选视觉摘要，明确标为 non-audio evidence；
- 严格 JSON schema 和对应 media family 的少量示例。

标注器输出 scene/entity/event 三层结构，并遵守：

1. `Speakers` 只列实际说话者；
2. speaker ID 连续，Content 中每个 speaker tag 都有实体定义；
3. verbatim speech/lyrics 不允许由 LLM 改写；
4. 每个事件必须有 span 和 evidence URI/score；
5. 低置信属性缺省，不用通用 persona 模板填空；
6. scene 描述只写持续条件，局部变化放事件；
7. 不可听内容允许输出 `unintelligible/unknown`。

### 4.6 Stage W5：多证据选择性验证

验证不是“多数模型投票后自动改写”，而是逐 claim 检查：

| claim | 主要证据 | 可选辅助证据 | 失败动作 |
|---|---|---|---|
| speech transcript | ASR + alignment + vocal stream | 第二 ASR | re-caption / human |
| speaker attribution | diarization + speaker embedding continuity | lip/face track | abstain / human |
| emotion/delivery | local vocal acoustics | text semantics | remove attribute |
| age/gender coarse tag | isolated vocal + calibrated verifier | VLM | flag，禁止自动补写 |
| SFX existence | local mixture/residual evidence | VLM synchronization | remove/flag |
| music/lyrics | music/vocal evidence + specialized teacher | metadata | lower specificity |
| environment/reverb | full mixture + acoustic estimator | visual scene | lower specificity |

所有 verifier 都需要独立校准集，至少记录：threshold、coverage、precision、abstention rate。没有校准集时只能产生 weak evidence，不能执行自动纠正。

### 4.7 Stage W6：人工审计

人工审计至少检查：

- transcript 与 speaker attribution；
- 漏掉的局部 SFX；
- 不可听却被写出的环境/persona；
- 稳定属性和局部情绪是否错层；
- separator artifacts 和 crowd leakage；
- 时间边界及不确定区间。

SwanTale 的 group-wise best–worst comparison 适合筛选“更丰富但不幻觉”的 caption：对同一音频生成四个有效候选，让标注员选最佳和最差，而不是要求在不同场景间维持一个绝对 1–5 分标准。但最终 benchmark 的事实与时间边界仍须逐项确认，不能只保留相对偏好。

## 5. 对模型与训练的具体借鉴

### 5.1 立即采用：角色分离的监督

在 MOSS/TAC-style baseline 中，先用特殊字段/标签区分：

- scene/global acoustic description；
- stable entity description；
- chronological local events；
- verbatim transcript/lyrics。

这相当于借鉴 SwanTale 的 caption/text/speaker-turn condition separation，但作用在输出端。模型仍可共享 audio encoder 和 LLM，不需要立即实现 MoE。

### 5.2 立即采用：可靠性课程

建议训练顺序：

1. **C0**：TAC-style 受控单源/少源数据，先学格式、类型和时间 token；
2. **C1**：真实 clean speech、music、SFX、lyrics 单域强标注，学语义与稳定属性；
3. **C2**：真实单源重混，学重叠与精确 source-derived spans；
4. **C3**：真实互联网 mixture 的高置信 pseudo-label，加入 scene/speaker 信息；
5. **C4**：local evidence supervision；
6. **C5**：CARC 的 add/remove/shift/nuisance consistency；
7. **C6**：只有 verifier 在独立人工集上校准后，才做 DPO/GRPO。

每进入下一阶段都要 replay 上一阶段的 anchor data，防止复杂 pseudo-label 破坏 ASR、时间格式和已有单域能力。SwanTale 在 GRPO 时不对难以可靠打分的 multi-speaker/audio samples 强行给 reward，而是用 supervised anchor replay 保留它们；这一点尤其值得照搬。

### 5.3 暂不采用：Unified MoE

SwanTale 的 MoE 用于异构波形生成，其 router 接收 diffusion timestep 与 frame latent。我们的 MOSS captioner 没有相同结构，也还没有证明共享 FFN 容量是瓶颈。现在实现 MoE 会混入：

- 额外参数和路由稳定性；
- expert collapse/load balance；
- 更多 GPU 和消融成本；
- 难以区分数据改进与架构改进。

只有在有效真实测试集上确认 speech/music/SFX 存在负迁移，并且共享模型弱于三个单域 oracle，才考虑 modality-aware adapters 或 MoE。

### 5.4 暂不采用：Engram

Engram 适合生成模型识别反复出现的稳定 caption phrase。我们的目标恰好需要防止模型依赖模板短语产生幻觉。只有真实 caption 词汇足够丰富、且错误分析证明长指令中的固定 marker 识别是瓶颈时，才测试 Engram/lexical memory。

当前更重要的是扩大真实文本和属性覆盖，并让 event-F1 真正依赖语义；旧数据每个类别只有少量固定句子，加入记忆模块只会加速模板记忆。

### 5.5 RL 的正确用法

SwanTale 的 GRPO 奖励可以启发我们，但不能直接复用。我们的候选 reward 应是：

- `r_schema`：严格格式和 cross-reference；
- `r_exist`：局部音频支持与 null calibration；
- `r_time`：对人工/强标注 boundary 的 collar/IoU；
- `r_text`：ASR/lyrics 的词错误或开放词汇语义一致；
- `r_speaker`：speaker attribution 与 embedding continuity；
- `r_cf`：remove/add/shift 后事件集合的差分正确性；
- `r_hallucination`：unsupported claim penalty。

RL 前必须满足：

1. 每个 reward 在独立人工集上报告与人类判断的相关性；
2. reward coverage 和 abstention 显式报告；
3. 对 multi-speaker/music/SFX 等难样本无可靠 reward 时只做 supervised replay；
4. 与等量 rejection sampling/DPO/SFT 对照，证明 RL 不是无必要复杂化。

## 6. 下一步实验的准确顺序

### Experiment D0：`B3-complex-v2` 数据锚点

目的：验证 renderer、split contract、replay、stems-sum、复杂度门禁和人工听感。

代码状态：**已具备，无需新增代码。**

执行入口：

```bash
bash scripts/run_b3_complex_v2_data.sh /cephfs/your_project/b3_complex_v2_74c5566
```

完整安装、烟测、监控、验收和回传方式见 [`docs/22_next_experiment_b3_complex_v2_data_anchor.md`](22_next_experiment_b3_complex_v2_data_anchor.md)。D0 不训练模型，也不评价 Swan-style scene/speaker caption。

通过条件：

- train/val/test 为 4000/500/500；
- replay、stems-sum、Ledger schema 全部通过；
- sample/source 三折完全隔离；
- 三折 `release` quality gate 全部通过；
- 固定人工试听没有模板级系统错误。

失败时只修失败的 renderer/sampler/contract 环节并重跑。不得降低阈值后继续训练。

### Experiment D1：200 条真实媒体 Swan-style annotation feasibility pilot

D0 通过后才启动。D1 不训练主模型，只回答：**我们的互联网音视频能否被可靠地转成 scene/entity/event 三层标注？**

建议从已爬取数据按下列主场景各取 25 条，总计 200 条；每条截取 10–30 s 且避免无关长尾：

1. 多说话人对话；
2. 重叠说话/打断；
3. speech + background music；
4. speech + ambience/local SFX；
5. speech + music + SFX；
6. singing/lyrics + music；
7. music + SFX/ambience、无 speech；
8. 强混响/回声/噪声/codec degradation。

数据划分：150 development + 50 人工强标 validation。50 条 validation 在开发 prompt/verifier 前冻结，不能反复查看后调规则。

#### D1 需要的输入

在写 adapter 代码前，需要服务器提供：

- 5–10 行脱敏后的原始媒体 manifest；
- 实际 audio/video 目录结构；
- 音频格式、采样率与平均时长；
- 是否已有 ASR、subtitle、platform metadata；
- 哪些媒体可研究使用、哪些可再分发；
- 服务器可使用的 GPU、模型缓存路径和离线限制。

没有这些信息时盲写下载/adapter 会再次制造不可运行代码。因此本轮不实现 D1 adapter；收到上述样例后应一次性实现 ingest、group split、teacher runner、schema validator 和 audit report，而不是逐模型写孤立脚本。

#### D1 最小输出

每条必须包含：

- source/license/group provenance；
- original mixture + optional vocal/residual URI；
- scene description + per-claim confidence/evidence；
- speaker entities + stable attributes；
- timestamped events + local delivery；
- teacher versions/checkpoint hashes；
- validation status：`verified | weak | abstain | rejected`；
- human corrections。

#### D1 预注册指标

在 50 条人工 validation 上报告：

| 维度 | 指标 |
|---|---|
| schema | validity、speaker ID/cross-reference error |
| existence | event precision/recall、unsupported claim rate |
| time | onset/offset MAE、F1@0.1/0.25/0.5/1.0 s |
| speech | cpWER/tcpWER、speaker attribution error |
| scene | environment claim precision 与 specificity |
| attributes | stable/local classification consistency、abstention coverage |
| evidence | evidence coverage、verifier precision/coverage curve |
| separation | artifact rate、vocal leakage、residual speech leakage |

建议 pilot 进入代码开发的最低门槛：

- schema/cross-reference validity 100%；
- 人工 event precision ≥ 85%，recall ≥ 75%；
- unsupported environment/speaker claims ≤ 10%；
- 所有自动属性修复的 precision ≥ 95%，否则改为只 flag；
- 高置信事件的 median onset/offset error ≤ 0.5 s；
- 50 条中无重复 source group 跨 development/validation。

这里的 0.5 s 是 D1 pseudo-label feasibility 门槛，不是论文最终宣称达到 0.1 s。只有强标 benchmark 才能检验真实 0.1 s 边界能力。

### Experiment D2：双视图 SFT 消融

D1 通过且 v0.3 schema 冻结后，再比较：

- A：event-only Ledger；
- B：scene + entity + event structured target；
- C：B + constrained global-caption rewrite；
- D：C + local evidence loss。

使用同一数据、split、steps 和 checkpoint，分别报告局部事件、全局环境、speaker attributes、文本、时间和 hallucination。不能只用一个 LLM judge 或 event-F1 决定优劣。

### Experiment D3：生成模型辅助的闭环压力测试

若 SwanTale 或同类统一生成模型未来公开，可将其用于**测试和数据扩增候选**：

```text
structured instruction
        ↓
audio generator
        ↓
SceneLedger captioner
        ↓
compare requested vs actually audible vs recovered
```

但 instruction 不是生成音频的 ground truth；生成器可能漏掉音效、speaker turn 或歌词。每条生成样本仍须经过音频侧 verifier/人工确认。最合适的用途是定向生成稀有组合和 counterfactual stress cases，而不是直接制造百万条伪真值。

## 7. 可直接使用和不可直接使用的组件

| 组件 | 可用性 | 本项目用法 |
|---|---|---|
| SwanTale/SwanVAE | 本次未找到官方代码/checkpoint | 仅借设计，不作为 base |
| SwanData-Caption | 内部约 70M caption records，未开放 | 复现其流程思想，不声称数据复现 |
| Seed-ASR 2.0 / Seed2.0 Lite | 服务/内部组件 | 用开放 ASR/audio-LLM 替代 |
| SwanAligner | 未找到公开实现 | WhisperX/MMS/MFA 等替代并独立校准 |
| SwanVerifier | 论文描述为内部过滤组件 | 用 WavLM/SenseVoice 等建立 selective verifier；不能声称复现 |
| 3D-Speaker | 开源 | multi-speaker diarization teacher |
| SenseVoice | 开源 checkpoint/代码 | ASR、emotion/AED 弱证据与交叉检查 |
| UVR/Demucs | 开源 | vocal/residual 辅助分离，保留原 mixture |
| DNSMOS/SQUIM | 有公开实现 | 质量元数据、过滤与分层，不作为 caption 真值 |

## 8. 论文贡献重新排序

结合 SwanTale 后，建议按以下逻辑写论文：

1. **数据**：真实 media-first、目标合成补长尾；分离/diarization/ASR/VLM 形成 evidence graph；selective verifier 和人工审计抑制系统性伪标签；
2. **表示**：Scene–Entity–Event 三层 Ledger，同时导出带 `<speech>/<music>/<lys>/<sfx>` 与时间戳的统一全局 caption；
3. **模型**：一次前向的 joint inverse scene parsing，track/event/local-evidence 明确绑定，而不是串联多个 captioner 后让 LLM 猜关系；
4. **训练**：可靠性课程 + anchor replay + CARC，RL 只优化经人工校准的可验证错误；
5. **推理**：证据约束、置信度和 abstention；全局重写不得增加 Ledger 外事实；
6. **评测**：存在性、语义、时间、speaker attribution、环境、属性、幻觉和声学鲁棒性分别报告。

论文中对 SwanTale 的定位应是：它证明细粒度 `Environment/Speakers/Content` 可以驱动统一复杂音频生成；我们研究相反方向，并进一步要求事件级时间与可审计证据。两者可以构成未来的 understanding–generation cycle，但不能把循环自身当真值。

## 9. 当前需要执行的动作

现在只做两件事：

1. 按 `docs/22` 在服务器运行 D0，并回传 `gate/`、`run.log`、data cards 和人工试听表；
2. 同时准备 D1 所需的 5–10 行脱敏媒体 manifest 与目录/授权说明。

D0 PASS 之前不提交新的 MOSS 训练。收到 D1 输入样例后，下一次代码工作将围绕一个完整、可恢复、带 provenance 的真实媒体 annotation pipeline 展开。

## 10. 主要来源

- [SwanTale 论文 v2](https://arxiv.org/html/2608.02023)
- [SwanAIGC 项目页](https://swanaigc.github.io/#swantale)
- [3D-Speaker 官方仓库](https://github.com/modelscope/3D-Speaker)
- [SenseVoice 官方仓库](https://github.com/QwenAudio/SenseVoice)
- [Ultimate Vocal Remover 官方仓库](https://github.com/Anjok07/ultimatevocalremovergui)
- [Microsoft WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
