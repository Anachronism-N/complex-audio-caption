# SceneLedger 设计澄清、实现边界与当前状态

> 更新日期：2026-08-11
>
> 本文逐项回答：输出为什么采用 Track–Event Ledger、scene graph 如何采样、source/stem/track/slot/event 的区别、query 数量、CARC 与 DPO、训练损失、Evidence-constrained Inference、粗到细定位、多证据验证，以及当前哪些内容已经实现。
>
> 状态标记：**[已实现]** 表示仓库已有代码；**[已验证]** 表示已有测试或实验产物；**[待实现]** 表示设计蓝图，不应写成论文实验结论。

## 0. 先给出结论

1. **最终用户仍然得到一个统一全局 caption。** Ledger 不是替代 caption，而是 caption 之前的规范化、可验证中间表示。推荐同时保留 `canonical_ledger` 和 `surface_caption` 两种输出。
2. **现有输出确实不够丰富。** 当前 renderer 给 source 配的文本非常通用，虽然 schema 已有 `conditions`、`attributes`、`relations` 等容器，但没有产生可靠的声学环境、说话人韵律/情感、音乐风格/结构标注。只做 LLM 重写不能恢复这些缺失事实。
3. **当前 B3-complex-v2 不是最终真实数据。** 它使用正弦、噪声、和弦等程序生成的占位波形，作用是验证采样、渲染、stems、标注、切分和门禁闭环。真实语料 `FileSourcePool` 只有基础接口，真实 source catalog 尚未接通。
4. **当前 `slot-aware` 不是 query 模型。** 它只是给自回归目标增加 `<n>` 和 `<slot>` 包装；真正的 track queries、event queries、Hungarian matching、track pointer 和 evidence mask 尚未实现。
5. **当前 CARC 只是删源数据增强。** 它在训练时随机去掉一条真实 stem，并用删去相应 track/event 后的文本继续做 SFT；尚未同时计算原始/干预样本之间的成对一致性损失，也不是 DPO。
6. **当前实际 loss 很简单。** 普通目标 token 用 CE，时间 token 用加权 CE。配置里的 `text_weight` 和 `type_weight` 目前没有形成独立 loss。后文列出的多项结构损失是分阶段实现计划，不应一次全部打开。
7. **Evidence-constrained Inference 既含训练所得能力，也含推理规则。** grammar/schema 约束与冻结外部 verifier 可以不再训练，但 track/event 置信度、局部证据头、细边界头必须训练或至少在 validation 上校准。
8. **粗到细定位不等于把完整模型运行两次。** 推荐一次 audio encoder 前向，先产生粗 proposal，再在同一特征图的局部窗口内细化到 10 Hz；只有超长音频才考虑第二次局部裁剪前向。
9. **视觉是可选的 privileged evidence，不是必要输入。** 没有视频时用 audio-only verifier；有视频时，视觉只能提出候选或增加置信，不能单独证明某个声音确实存在。
10. **当前最紧急的不是继续加 loss。** 应先在服务器完整运行 B3-complex-v2 数据闭环、人工试听并冻结无泄漏基线；随后接入真实 source catalog，再开始 query 架构实验。

## 1. 为什么这样定义输出；能否得到统一全局 caption

### 1.1 设计 Ledger 不是因为“数据容易构造”

复杂音频的一句话描述同时包含四类不同事实：

- **存在性**：是否真的有某个声音；
- **归属**：它来自哪个说话人、歌手、乐器或其他声源；
- **时间**：何时开始、结束、重复或重叠；
- **语言表现**：用什么自然语言把事实描述出来。

如果直接训练一个长文本 $Y$，这些变量全被压在 $p(Y\mid x)$ 中。文本读起来流畅，不代表每个短语都能在音频里找到证据。Ledger 把中间事实显式化：

$$
L=(C,\mathcal{T},\mathcal{E},\mathcal{R}),
$$

其中 $C$ 是全局声学条件，$\mathcal{T}$ 是持续声源集合，$\mathcal{E}$ 是带时间的可描述事件集合，$\mathcal{R}$ 是重叠、先后、打断、回声等关系。这样可以分别评价 source count、事件存在、track 归属、边界、文本和全局描述，而不是只用一个语言指标混在一起。

因此，Ledger 的主要价值是**可训练、可约束、可评测、可追溯**；合成数据能够方便地产生它只是附带优势。

### 1.2 推荐的两层输出

第一层是信息尽可能完整的规范表示：

```json
{
  "scene": {
    "global_summary": "室内访谈片段，背景有轻柔爵士乐并伴随明显混响。",
    "acoustic_environment": {
      "space": "medium_room",
      "reverberation": "strong",
      "echo": false,
      "background_noise": "low crowd murmur",
      "confidence": 0.82
    }
  },
  "tracks": [
    {
      "id": "T1",
      "kind": "speech",
      "identity": "S1",
      "spans": [[0.7, 2.9], [6.2, 7.4]],
      "attributes": {
        "voice": "low-pitched and slightly hoarse",
        "prosody": "fast",
        "emotion": "excited",
        "emotion_confidence": 0.68
      }
    },
    {
      "id": "T2",
      "kind": "music",
      "spans": [[0.0, 12.8]],
      "attributes": {
        "genre": "jazz",
        "instruments": ["piano", "upright bass"],
        "tempo_bpm": 108,
        "structure": "drums become stronger after 6.0 s"
      }
    }
  ],
  "events": [
    {
      "id": "E1",
      "track_id": "T1",
      "type": "speech",
      "spans": [[0.7, 2.9]],
      "content": "我们现在开始。",
      "attributes": {"delivery": "quickly"},
      "confidence": 0.94
    }
  ],
  "relations": [
    {"subject": "E1", "predicate": "overlaps", "object": "T2"}
  ]
}
```

第二层是面向用户的统一 caption，由确定性 serializer 或受限 LLM 生成：

```xml
<scene t="0.0-12.8">室内访谈片段带有明显混响和低声人群背景。</scene>
<music track="T2" t="0.0-12.8">约 108 BPM 的轻柔爵士乐持续播放，以钢琴和低音提琴为主，6.0 秒后鼓点增强。</music>
<speech track="T1" speaker="S1" t="0.7-2.9" manner="快速、兴奋">“我们现在开始。”</speech>
```

因此，最初希望的“一次输入音频，直接得到一个全局统一 caption”完全保留。模型内部可以先预测 Ledger，接口层再返回一条 caption；用户不需要手工调用四个系统。

### 1.3 为什么仅重写当前格式仍会丢信息

LLM 只能改写输入中已有的事实。当前合成 source 文本类似“一名说话者正在讲话”“背景音乐持续播放”，没有标注：

- 房间大小、混响强度、回声、噪声类型与 codec；
- 声音的距离、方位、前景/背景和遮蔽程度；
- 说话速度、音高、音色、情感和其他副语言信息；
- 音乐 genre、mood、instrument、tempo、key、section、演奏变化；
- 事件间的打断、回应、因果、echo-of 等关系。

所以“重写现有 `<speech>/<music>/<sfx>` 文本”只能改善文风，不能补回未进入 Ledger 的信息。正确做法是扩充监督来源和规范字段，再让 rewriter 只组合这些已验证事实。对说话人属性应描述可听见的音色、音高和韵律，避免把不可靠的性别、年龄或身份推断写成事实。

### 1.4 对当前 schema 的具体决定

现有 [`schema.py`](../src/sceneledger/data/schema.py) 已包含 `conditions`、track/event `attributes`、`relations` 和 `evidence`，所以不必推翻 Track–Event 结构；下一版 schema 应：

1. 增加 `scene.global_summary` 与更细的 `acoustic_environment`；
2. 为不同 track/event type 规定类型化 attributes，而不是无限制自由字典；
3. 所有主观属性保存 `confidence` 和 `provenance`；
4. 区分物理真值、teacher 预测和人类描述；
5. 把 `surface_caption` 视为 Ledger 的派生视图，而非训练真值的唯一载体。

## 2. scene graph 是怎样采样和混合的

### 2.1 当前代码的完整流程

当前入口是 [`render.py`](../src/sceneledger/cli/render.py)，规则在 [`scene_graph_sampler.py`](../src/sceneledger/data/scene_graph_sampler.py)，DSP 与标注在 [`renderer.py`](../src/sceneledger/data/renderer.py)。每条数据按以下顺序生成：

```text
template weighted sampling
  -> scene duration
  -> source types/count dictated by template
  -> pick one source key for each type
  -> sample onset/gain/repeat/RIR/echo
  -> load or synthesize each source waveform
  -> background loop + gain + fade + repeat + per-source RIR
  -> place every source on the scene timeline (one placed stem per source)
  -> sum all placed stems into dry mixture
  -> apply scene-level echo and clipping guard
  -> compute per-stem 100 ms activity
  -> build tracks/events/spans
  -> save mixture, stems, manifest and hashes
```

模板不是自然语言生成，而是确定性规则。例如：

| template | source 组成 | 特殊规则 |
|---|---|---|
| `speech_over_music` | music + speech | music 从 0 s 开始并循环，speech 随机放置 |
| `speech_music_sfx` | music + speech + sfx | 三类可重叠 |
| `lyrics_over_music` | music + vocal | vocal 标成 `<lys>` |
| `speech_music_lyrics_sfx` | music + speech + vocal + sfx | 当前最复杂的四源模板 |
| `repeated_event` | ambience + repeated sfx | SFX 重复 2–5 次并分散到时间轴 |
| `overlapping_speakers` | speech S1 + speech S2 | 两个 onset 被约束到邻近位置 |
| `isolated_sfx` | one sfx | 只用于约 3% 的稀疏诊断样本，时长 2–5 s |

模板通过配置中的 `template_weights` 加权随机选择；每个 scene 的 seed 决定 duration、source key、onset、gain 和退化参数，因此同一 manifest 可重放。

### 2.2 当前是不是使用语料库数据

**不是。** `b3_complex_v2_{train,val,test}.yaml` 明确设置 `pool.kind: synthetic`。当前 source 是程序产生的 formant-like speech、sustained vocal、和弦、瞬态噪声和 ambience，占位 key 类似 `speech:023`。这些声音只用于验证工程闭环，不能代表真实语音、真实歌词、真实音乐风格或互联网复杂场景。

仓库已有 `FileSourcePool(by_kind=...)`，能够从文件列表中采样真实音频，但还缺少论文训练真正需要的：

- 真实 source catalog 文件及稳定 schema；
- 数据下载、许可、去重、质量检查与 recording-level split 的完整闭环；
- source 自身的 transcript、lyrics、音乐属性和 SFX caption；
- 防止同一 speaker/song/recording 跨 split 的 group identity；
- 对 clip 边界、静音、响度和污染源的审计。

因此，当前数据管线可以称为“renderer 与数据合同已实现”，不能称为“真实 TAC 数据流程已完整复现”。

### 2.3 当前混合规则还缺什么

当前 v2 已实现 synthetic RIR、scene-level echo、前后景 gain 和 overlap；但是：

- `noise_snr_db` 当前固定为 `None`，尚未真正叠加独立噪声；
- `codec` 当前固定为 `None`，尚未做压缩退化；
- synthetic RIR 不是实测房间脉冲响应；
- source 语义和声学属性非常贫乏；
- scene-level echo 在 stems activity 生成之后施加，当前标签没有显式把 echo 尾音建成独立 span 或 `echo_of` 关系；
- 模板分布是工程先验，不是从 Bilibili/Instagram/TikTok 估计的真实联合分布。

这些限制决定了 v2 只能作为数据单元测试和无泄漏实验锚点。

### 2.4 LLM 选择混合内容是否更好

**LLM-only 不更好，规则-only 也不应是最终方案。推荐“统计先验 + 受约束 LLM proposal + 确定性 renderer”。**

LLM 的优势是能提出语义上自然的组合，例如“街头采访常有车辆、人群与远处音乐”，并给出合理关系；缺点是不可复现、容易偏向有故事性的声音、无法保证 source 存在、可听性、许可、split 隔离和目标分布。

最终数据生成建议分四步：

1. 从真实互联网样本的只读分析中统计 scene type、source count、overlap、event density、SNR、T60 和组合共现，得到经验先验 $p_{real}(G)$；
2. LLM 只输出满足 JSON Schema 的高层 scene proposal，包括 source 类型、语义关系和大致时间安排；
3. catalog resolver 把 proposal 中每个节点绑定到真实且 split-safe 的 source，无法解析则拒绝该 proposal；
4. renderer 负责精确时间、增益、RIR、噪声和 stems，quality gate 再做拒绝采样。

第一篇可复现实验仍应以规则模板为主，LLM sampler 作为后续 data-diversity ablation：固定相同 source pool 和样本量，比较 `rule-only`、`empirical-prior`、`LLM-constrained` 在真实人工集上的分布距离和模型性能。不能用 LLM judge 自己证明 LLM sampler 更真实。

## 3. source、stem、track、event 和 slot 分别是什么

| 术语 | 本项目中的操作性定义 | 例子 | 是否是 waveform |
|---|---|---|---|
| source | 一个物理或感知上的发声来源，也是混音前 catalog 中的一条素材 | 某位说话人的一句干声、狗叫素材、一段钢琴伴奏 | 通常是 |
| placed stem | 一个 source 经 gain、repeat、RIR 后，被放进完整 scene 时间轴的单独贡献 | 20 s 场景中只在 4.6–4.9 s 非零的玻璃破碎轨 | 是 |
| mixture | 所有 placed stems 求和并施加 scene-level 退化后的最终输入 | speech + music + sfx | 是 |
| track | 同一持续身份/来源在当前 scene 中的逻辑通道 | 说话人 S1 两次讲话仍属于 T1 | 不一定；学生中可为隐变量 |
| event | track 产生的最小可描述单元 | S1 的第一句话、一次玻璃破碎、歌曲副歌段 | 不一定 |
| slot/query | 神经网络中一个固定容量、可学习的输出位置 | 第 3 个 track query 最终绑定 T1；空 query 预测 null | 否 |

另外要区分：`source_id` 是数据构造时的 source 实例标识；`track_id` 是一个 scene 内的逻辑标识；`slot index` 只是模型 tensor 的位置，不能跨样本解释成固定声源类别。

### 3.1 怎样从 stems 自动得到“精确标注”

由于混音前知道每条 placed stem $s_j(t)$，可以在 100 ms 帧上计算 RMS activity：

$$
a_{j,k}=\mathbb{1}[\operatorname{RMS}(s_j[k])>\theta_j].
$$

相邻 active frames 按 merge threshold 合并成 spans。每条 stem 自然对应一个 track；speech/lyrics 的多个活动段可以生成多个 utterance/line events，同一 repeated SFX 的多个活动段可保存在一个 multi-span event 中。因为 onset、repeat gap 和 waveform 都由 renderer 知道，所以 attribution 和时间比从 mixture 反推更可靠。

但“精确”有严格边界：

- **精确的是混音操作、stem 归属和程序定义的 activity。**
- **语义文本只与 source 原始标注一样准确。** 输入只标为 `dog bark`，renderer 不会自动知道“远处、急促、两只狗”。
- RMS threshold、RIR 尾音、fade 和人类感知 onset 会引入边界定义差异；0.1 s 是标签网格，不自动等于 0.1 s 人类精度。
- 当前 scene-level echo 没有反馈到 stem activity，因此它不能给 echo 本身提供完整真值。

## 4. Track–Event 建模本质、隐式性、数量与可行性

### 4.1 对任务本质的更准确表述

可以近似概括为：

```text
mixture
 -> 找到持续的感知来源（tracks）
 -> 找到各来源的活动区域
 -> 把活动区域组织成可描述单元（events）
 -> 识别/转写每个 event 的内容与属性
 -> 组合成统一 caption
```

但 track 不一定等同于可完美分离的物理 waveform。单通道强混响场景可能没有唯一分解，模型只需给出对 caption 有用的“感知来源 lane”。例如：

- 同一个 speaker 的两句话：1 track，2 events；
- 一个 glass-break source 重复三次：1 track，1 multi-span event，或按评价定义拆成 3 instance events；
- 一首歌：accompaniment 是 music track，主唱是 vocal track，多句歌词是多个 lyric events；
- diffuse crowd noise：可归到 ambience/residual track，而不强行拆成十个人。

track/event 的操作性定义必须写进 annotation guide，否则“一个鼓组算一个 track 还是多个乐器 track”本身没有唯一答案。

### 4.2 哪些显式，哪些隐式

| 阶段 | track/event 状态 |
|---|---|
| 程序混音 | 显式真值：source、placed stem、activity、track、event 均可追溯 |
| 有真实 stems 的多轨语料 | track/stem 显式；event 由歌词、ASR、MIR、activity 等标注 |
| 无标注互联网音视频教师流程 | pseudo-explicit：separator/diarizer/SED 给候选 track/event 和置信度 |
| 最终学生训练 | 用真值或伪标签监督内部 track/event queries |
| 最终学生推理 | 隐式分解：直接从 mixture 输出 Ledger，不要求导出可播放 stems |

当前仓库只完成前两列中的“程序混音显式标签”以及自回归 caption baseline。真正隐式 query 学生尚未实现。

### 4.3 未知数量如何处理

第一版不需要动态创建 tensor。设最大容量 $K_t$ 个 track queries、$K_e$ 个 event queries，每个 query 额外预测 `presence/null`：

$$
\hat N_t=\sum_{i=1}^{K_t}\mathbb{1}[p_i^{track}>\tau_t],\qquad
\hat N_e=\sum_{j=1}^{K_e}\mathbb{1}[p_j^{event}>\tau_e].
$$

训练时用 Hungarian matching 把无序 prediction 和真值配对；未匹配 queries 学习 `null`。所以 $K$ 是容量上限，不是强制输出数量。固定上限加 null 的思路来自 [DETR](https://arxiv.org/abs/2005.12872) 的 set prediction；[SEDT](https://arxiv.org/abs/2110.02011) 已把 1D-DETR/event query 思路用于 sound event detection。

`K_t=8, K_e=24` 目前只是设计文档中的候选值，不应直接冻结。正确选择方法是：

1. 在通过质量门禁的 train 和真实开发集上统计同时/累计 track 数与 event 数；
2. 令容量覆盖 train 的 P99 或最大值，并预留约 20% null slack；
3. 对 `Kt={4,8,12}`、`Ke={12,24,36}` 做容量消融；
4. 报告 truncation rate、null occupancy、duplicate rate、source-count MAE 和显存/延迟；
5. 对超过容量的样本采用分窗或 EEND-EDA 式迭代，而不是静默丢弃。

[EEND-EDA](https://arxiv.org/abs/2106.10654) 证明了 diarization 中可以用 attractor 与 stopping condition 处理未知说话人数；[Slot Attention](https://arxiv.org/abs/2006.15055) 说明交换对称的 slots 可从感知特征中竞争性绑定实体；[AudioSlots](https://arxiv.org/abs/2305.05591) 已给出 slot-centric blind audio separation 的 proof of concept。它们支持“用可交换 slots 表示未知音频组成”这一方向的可行性，但不证明 SceneLedger 在统一 caption 上必然有效。

### 4.4 如何证明本设计有效，而不是只靠类比

论文需要至少完成以下证据链：

1. **Oracle-track 上限**：用真 track/activity 喂给 event text decoder；若仍不比 flat caption 好，track 分解不是当前瓶颈；
2. **Predicted-track 实验**：替换 oracle track，报告性能下降，量化 track detector 的实际贡献与误差传播；
3. **Flat vs event-only vs track–event**：同 encoder、参数量、数据和训练步数，仅改变结构；
4. **source-count scaling**：按并发源数 1/2/3/4+ 分桶，检验结构优势是否只在复杂场景出现；
5. **identity continuity**：同 speaker/singer 多段出现时测 track association 和 speaker-attributed WER；
6. **重复与重叠**：测 multi-span event、重叠区 segment F1、duplicate/omission；
7. **query 容量消融**：证明结果不依赖幸运选择的 $K_t/K_e$；
8. **真实人工集验证**：合成集上的 set prediction 成功不能替代真实复杂声景结果。

## 5. 声源级反事实干预在哪一层；与 DPO 的区别

### 5.1 完整 CARC 同时涉及数据层和训练层

数据层产生带同一 group ID 的干预组：

$$
\{x,\ x^{-e},\ x^{+e},\ x^{\tau e},\ a(x)\},
$$

分别表示原始、移除一个 source、加入 source、时间平移 source，以及施加不改变可听事件身份的噪声/RIR/codec 视图。数据记录被干预的 `source_id`、track/event 映射、shift 和可听性。

训练层使用已知干预关系约束：

- removal：目标 source 的 track/events 应消失；
- retention：其他 track/events 应保持；
- shift：目标 events 的时间按 $\tau$ 平移，语义和身份保持；
- nuisance：事件仍可听时语义/归属不变，不可听时应降低 confidence 或 abstain。

### 5.2 当前代码实际做了什么

[`train_carc.py`](../scripts/train_carc.py) 在每个训练 step 以一定概率：

1. 随机选择一个 source；
2. 把其余 stems 相加得到删源 mixture；
3. 从目标 Ledger 删除对应 track/events；
4. 对这个新 `(audio, target)` 做与普通样本相同的 token CE。

它没有在同一个 batch 同时前向 $x$ 与 $x^{-e}$，没有做 prediction matching、retention consistency、time-shift equivariance 或 audibility gating。因此当前实现更准确的名字是 **source-removal SFT augmentation**，只是完整 CARC 的最小原型。

### 5.3 与 DPO 的本质差别

| 维度 | 完整 CARC | DPO |
|---|---|---|
| 监督来源 | 已知的音频物理干预与集合变化 | 同一 prompt 下 chosen/rejected 输出偏好 |
| 学习对象 | track/event 存在、归属、时间、语义的等变/不变关系 | 提高 chosen 相对 rejected 的策略概率 |
| 是否需要 reference policy | 不需要 | 标准 DPO 需要 |
| 是否天然包含时间代数 | 是，shift 直接给出时间变化 | 否，除非偏好数据和输出中显式编码 |
| 是否是 RL | 不是 | 也不是在线 RL；是偏好分类式目标 |

[DPO](https://arxiv.org/abs/2305.18290) 把偏好优化写成相对 reference policy 的分类式损失。它可在后期用于“有证据回答优于幻觉回答”的 preference alignment，但不能替代 CARC 的物理干预监督。当前仓库没有 DPO。

## 6. 当前 loss、完整训练 loss 与局部证据学习

### 6.1 当前真正运行的 loss

当前 B3/slot-aware 使用 MOSS-Audio + LoRA 的 causal language modeling：

$$
L_{token}=\frac{1}{M}\sum_{m=1}^{M}w(y_m)\operatorname{CE}(p_m,y_m),
$$

其中普通 token 的 $w=1$，atomic timestamp token 的默认 $w=5$。`<n>`、`<slot>`、type tag 和文本都只是目标序列中的 token，并没有独立分类头。

必须特别注意：

- `configs/model/*.yaml` 中虽有 `text_weight` 和 `type_weight`，当前训练入口实际只读取 `timestamp_weight`；
- slot-aware 当前只改变目标字符串，不是 permutation-invariant slot loss；
- CARC 原型仍然计算同一个 token loss；
- 普通 `train.py` 有 cosine schedule，当前 `train_carc.py` 没有真正应用配置中的 scheduler；
- 配置中的 `steps` 当前按 micro-sample 递增，gradient accumulation 后的 optimizer update 数更少。

这些是解释当前 CARC 负结果时应优先排查的实现因素，而不是继续叠加新 loss。

### 6.2 完整 query 模型建议的最小损失

第一版 query 模型只实现可诊断的五项：

$$
L_{stage1}=L_{presence/type}+\lambda_a L_{activity}
+\lambda_b L_{boundary}+\lambda_p L_{pointer}+\lambda_{txt}L_{text}.
$$

- `L_presence/type`：matched query 的 type CE 与 unmatched query 的 null/focal loss；
- `L_activity`：100 ms mask 的 BCE + Dice；
- `L_boundary`：onset/offset 分布 NLL，允许边界不确定区间；
- `L_pointer`：event 指向 matched track 的 CE；
- `L_text`：只在 matched event 上计算的 token CE。

`containment`、local evidence、CARC 和 calibration 不应在第一天全部打开。推荐课程：

1. exact-stem 合成数据上只训练结构头，确认 count/type/activity/pointer 收敛；
2. 加入 event text，确认结构指标没有崩；
3. 加入小权重 local evidence，对比 hallucination 与 text quality；
4. 加入完整 paired CARC，只看反事实 challenge set 是否改善；
5. 最后混入低权重真实 pseudo-ledger，并做 confidence calibration。

每一步保留上一阶段 checkpoint，记录每项 loss 的量级和共享 encoder 梯度范数。若两个目标的梯度长期冲突，再考虑 GradNorm、uncertainty weighting 或分 adapter；不应一开始依赖复杂自动加权掩盖监督错误。

### 6.3 什么是“局部证据学习”

对 event $e_i$，用预测/真值 activity $a_{it}$ 与 track mask 从 audio features $H_t$ 池化局部表示：

$$
h_i=\frac{\sum_t a_{it}H_t}{\sum_t a_{it}+\epsilon}.
$$

再将 event 文本编码为 $g(c_i)$，做 scene 内对比：

$$
L_{local}=-\log
\frac{\exp(\operatorname{sim}(h_i,g(c_i))/\tau)}
{\exp(\operatorname{sim}(h_i,g(c_i))/\tau)+
\sum_{j\in\mathcal N_i}\exp(\operatorname{sim}(h_i,g(c_j))/\tau)}.
$$

负例优先选择同一 scene 的其他 track、相邻但无该 source 的时间段、source-removed residual 和易混 sibling。再加入局部-外部 margin：

$$
L_{outside}=\max(0,m-operatorname{sim}(h_i,g(c_i))+
\operatorname{sim}(h_i^{outside},g(c_i))).
$$

直观上，“glass breaking”必须在玻璃破碎 stem 活跃的局部比在其余时间更匹配。对同时发生的事件，仅用时间 crop 会混入其他 source，因此 exact stems 上用 stem-local feature，预测阶段用 track TF mask 或 separator teacher mask。

局部证据学习不是让另一个 LLM 判断句子是否合理；它约束文本 embedding 与对应声学区域的相似性。它也有风险：弱 audio-text encoder 可能只识别粗类别而忽略“远处、金属、快速”等属性，所以必须按粗事件与细属性分别评价。

## 7. Evidence-constrained Inference 与结构化 Ledger 解码

### 7.1 哪些无需训练，哪些需要训练

| 组成 | 是否需要本项目训练 |
|---|---|
| 标签 grammar、时间合法性、ID 引用和 JSON Schema 校验 | 不需要 |
| 确定性排序、去重、serializer、失败回退 | 不需要 |
| 冻结的 WhisperX/FLAM/Demucs/CLAP 等外部 verifier | 可不训练，但阈值需在 dev 校准 |
| track/event presence、activity、pointer、confidence | 需要训练 |
| coarse-to-fine boundary refinement | 需要训练 |
| 学习式 evidence reranker/corrector | 需要训练；第一版可不用 |

所以 Evidence-constrained Inference 不是一个纯后处理名称，也不是一定要另训一个巨大模型。第一版可以是“已训练的结构头 + 无训练 grammar + 冻结 audio verifier”。

### 7.2 “解码”到底是什么

神经网络输出的是 logits、概率、mask 和 token 分布，不是天然合法的 Ledger。结构化解码负责：

1. 从 $K_t/K_e$ 个 queries 中选出非 null 项；
2. 将 activity 概率转成一个或多个 spans；
3. 从 boundary 分布得到 onset/offset 与 uncertainty；
4. 解析 event-to-track pointer，拒绝指向 null track 的 event；
5. 合并重复 event，保留真正的 multi-span 重复；
6. 强制时间在音频范围内、span 有序、ID 唯一、标签闭合；
7. 调用 verifier 后执行 accept、降置信、删除或 abstain；
8. 生成 canonical Ledger，再确定性序列化或受限改写成一个 caption。

这一步需要“解码”，因为直接对最大概率 token 做 greedy generation 可能出现数量不一致、时间逆序、未闭合标签、重复事件或 event 指向不存在 track。

### 7.3 LLM 在该阶段的边界

LLM 只做 surface realization：在锁定 event IDs、spans、type、track pointer 和 transcript/lyrics 后，改善句子衔接并描述已验证关系。重写后重新解析；一旦增加 event、改时间或猜不可听词，直接回退确定性 serializer。论文主指标计算 canonical Ledger，而不是 LLM 润色文本。

## 8. 粗到细时间定位是否需要二次推理

推荐实现是**一次 encoder，两个尺度的顺序 head**：

```text
audio -> shared encoder features
      -> coarse head (0.5 s or 0.25 s event proposal)
      -> gather local multi-scale features around each proposal
      -> fine boundary/activity head (10 Hz + continuous residual offset)
      -> quantize final display time to 0.1 s
```

这在计算图内部是两阶段，但不需要完整 MOSS/LLM 对原音频运行两次。coarse head 解决长时间搜索，fine head 只处理候选附近，降低 10 Hz 全局 query 的优化难度。文本在边界确定后只解码一次。

可选的第二次音频前向只用于：

- 数分钟以上长音频，需要先分窗再对候选局部高分辨率编码；
- 原 encoder 时间分辨率明显不足，需要从原 waveform 裁剪窗口给专用 fine encoder；
- 离线高精度模式允许额外延迟。

论文必须同时报告一次前向版本和可选 refinement 的延迟/显存，且把“0.1 s 输出网格”与 onset/offset MAE 分开。

## 9. 多证据验证使用什么模型；没有视觉怎么办

### 9.1 证据从哪里来

没有 ground truth 的推理阶段无法“证明”一个 caption，只能用相互独立的声学观测检查一致性。证据包括：

- 主模型内部：track presence、activity、TF mask、event confidence、local feature；
- event 局部音频：预测 span 的 crop 与 span 外对照区域；
- 显式分离：target stem 与 residual，检查 target/residual margin；
- 独立专家：ASR、diarization、singing voice、MIR、open-vocabulary SED/audio-text model；
- 可选视频：VLM 候选、mouth/object motion、audio-visual synchronization。

### 9.2 按 event 类型的第一版 verifier

| event | 可用模型/工具 | 验证逻辑 |
|---|---|---|
| speech | WhisperX 或 MOSS ASR + pyannote diarization/VAD | transcript 置信、局部词时间、speaker activity 与 span 是否一致 |
| lys | Demucs vocals + singing ASR/MOSS | vocal stem 中是否有可辨演唱；不可辨时只保留 `unclear vocals`，不猜歌词 |
| music | Demucs accompaniment + music tagger/MIR | genre/instrument/tempo 是否在局部稳定支持，结构变化是否与时间一致 |
| sfx | FLAM/open-vocabulary SED 或独立 audio-text encoder | caption 在 event-local 的分数高于 outside/residual；检查 sibling confusion |
| acoustic environment | blind RIR/T60、noise/codec classifier 或人工标注 | 只输出可校准属性，低置信时使用“可能/明显”等级或省略 |

可以定义可解释分数，而不是让一个 judge 给总分：

$$
S(e)=\alpha S_{presence}+\beta S_{local-text}
-\gamma S_{outside-text}+\delta S_{expert-agree}
+\eta S_{target-residual}.
$$

阈值在冻结 dev 上选择，输出三态：`accept`、`low-confidence/abstain`、`reject`。第一版 correction 只允许删除、降低细节层级、修正到专家共同支持的候选或微调边界；不允许 verifier 自由生成新事件。

### 9.3 是否必须引入视觉

不必须。论文主任务和主模型应是 audio-only，否则没有视频的部署场景无法使用，也难判断收益来自声音理解还是画面猜测。

有视频时增加 `AV-verified` 模式：VLM 提出可见 source/action，AV-sync 检查画面动作与声音是否同步，视觉支持只作为额外证据。狗在画面中但没有狗叫、乐器可见但没有演奏时，视觉不能把该事件写入 Ledger。训练时视频可以是 privileged modality，推理主模型仍只用音频；这也便于后续扩展音视频 caption。

## 10. 当前已经推进/验证了什么，哪些只是构想

### 10.1 状态总表

| 模块 | 状态 | 当前证据 | 不能宣称的内容 |
|---|---|---|---|
| Ledger v0.2 schema/parser/serializer | **[已实现]** | Pydantic schema、0.1 s span、track/event 引用与验证 | 丰富声学/情感/音乐属性已被可靠标注 |
| TAC-style synthetic sampler/renderer | **[已实现]** | 9 类模板、gain/onset/repeat/RIR/echo、mixture/stems/manifest | 使用了真实语料或匹配真实互联网分布 |
| B3-complex-v2 分布修复 | **[已实现][局部验证]** | 背景循环、稀疏样本限额、重复 SFX、多说话重叠、split contract 和 fail-closed gate；CPU smoke/unit tests 通过 | 完整 4k/500/500 数据已在服务器生成并人工验收 |
| 旧 B3-5k 审计 | **[已验证]** | 新 release gate 能拒绝旧数据；旧数据含大量长静音/稀疏场景 | 旧 B3-5k 是有效论文数据 |
| MOSS + LoRA + atomic timestamp baseline | **[已实现][探索性实验]** | 可训练/推理/评价；时间 token 加权 CE | TAC exact reproduction 或真实 0.1 s 精度 |
| slot-aware text format | **[已实现][探索性实验]** | `<n>` + `<slot>` 在旧合成评测上优于 flat B3 | 已实现 permutation-invariant slots/queries |
| source-removal CARC prototype | **[已实现][探索性实验]** | 训练时删 stem 和相应 Ledger 项 | 已实现 paired CARC、一致性 loss 或已证明有效 |
| P5 Demucs 分轨 topline | **[探索性实现]** | 可按预测 stems 分别 caption | stems 的固定类别映射可作为真实 track 真值 |
| 真实 source catalog/D0 | **[待实现]** | 只有 `FileSourcePool` 基础接口 | 真实数据流程已经复现 |
| track/event query 学生 | **[待实现]** | 仅有设计与相关工作依据 | track/event queries 已训练或有效 |
| local evidence/TF mask | **[待实现]** | 仅有 loss 和架构设计 | 已降低真实幻觉 |
| 完整 paired CARC | **[待实现]** | 仅有 intervention algebra | 优于强 baseline |
| coarse-to-fine boundary head | **[待实现]** | 仅有方案 | 100 ms 人类边界精度 |
| Evidence-constrained decoder/verifier | **[待实现]** | schema 校验部件已有，语义 verifier 尚无闭环 | 多证据验证已经纠正真实 caption |
| 受限 LLM/VLM 重写与验证 | **[待实现]** | 仅有接口原则 | LLM/VLM 能作为 ground truth |
| WildMix-Cap 人工 benchmark | **[待实现]** | 有 annotation/data 设计 | 已有真实无泄漏 test set |

### 10.2 已有数值如何解释

仓库已有旧数据上的探索性结果：

| 模型 | Event-F1 | hallucination | omission |
|---|---:|---:|---:|
| B3 | 0.902 | 198 | 140 |
| B3 slot-aware | 0.926 | 99 | 87 |
| B3 slot-aware-5k | **0.948** | **46** | **54** |
| B3 CARC | 0.912 | 57 | 118 |
| B3 CARC-5k | 0.920 | 67 | 94 |

这些数字只能用于工程诊断：旧 500 条评测与训练 fold 高度重叠，且数据分布已被新 gate 判定不合格。最新 CARC-5k 还显示，删除增强相对 slot-aware-5k 使 F1 下降 2.8 个百分点、hallucination 增加 46%、omission 增加 74%。因此当前证据不支持“CARC 已经有效”；更合理的结论是，现有 unpaired removal augmentation 在较强 5k baseline 上有害，需要在有效数据、严格 test 和专门反事实 challenge set 上重新验证完整目标。

### 10.3 当前数据流程的真实完成度

```text
程序占位 source -> scene graph -> renderer -> exact stems/time -> manifest
                  -> source-disjoint split contract -> quality gate
```

这条**代码链路已完成并通过小规模 CPU 检查**。但下面这条论文需要的链路尚未完成：

```text
真实合规语料下载/整理 -> source catalog -> recording/speaker/song-level split
 -> 真实属性标注 -> realistic scene prior -> full render
 -> 自动 gate -> 人工试听 -> frozen train/dev/test
```

所以当前最准确的进度判断是：**数据生成器和实验合同基本就绪，真实训练数据尚未准备好，新的无泄漏 v2 全量基线也尚未产生。**

## 11. 接下来的实施顺序与停止条件

### M0：先完成数据锚点，不加新模型

1. 在服务器运行 [`20_valid_data_protocol_v2.md`](20_valid_data_protocol_v2.md) 中的 4000/500/500 全量生成；
2. 确认 `experiment_data_summary.pass=true`、source leakage 为 0；
3. 按 template、最低 active ratio、最大尾静音、最低 overlap、最大 T60 分层试听；
4. 冻结 dataset ID、manifest hashes、test references；
5. 只在冻结 test 上跑 slot-aware baseline。

若这一阶段失败，停止 GPU 训练并修数据，不讨论 query、CARC、RL 或 verifier。

### M1：接入真实 source catalog

按 speech、vocal、music、sfx、ambience 建 catalog，至少记录路径、hash、license、recording/group ID、duration、caption/transcript/lyrics、attribute provenance。混音前按 recording、speaker、song 和 fingerprint group 切分。先做 1k real-source controlled mixtures，通过人工试听后再扩展。

### M2：补足“丰富描述”监督

扩 schema 与 source annotations，先完成可客观验证的字段：noise/RIR/codec、distance、foreground/background、speech rate、vocal presence、instrument、tempo 和 music section。情感、音色和 genre 保存 annotator/teacher agreement 与置信度，不把主观标签当物理真值。

### M3：建立严格基线

在同一冻结数据上比较 atomic、slot-aware text format 与 oracle-stem cascade。先确认无泄漏 baseline 可复现，并报告不同 source count/SNR/T60/overlap 桶。

### M4：实现最小 track–event query 模型

只实现 presence/type/activity/pointer/boundary/text 五项；先做 oracle-track 与 predicted-track 两组。若 predicted track–event 不在 3+ source、overlap 和 repeated-event 桶显著优于 flat baseline，停止扩展 local evidence。

### M5：再实现 local evidence 与完整 paired CARC

local evidence 必须在 source-removal、sibling confusion 和 out-of-span challenge 上降低 hallucination；完整 CARC 必须同时改善 deletion accuracy 与 retained-event consistency，且不能显著增加 omission。否则不进入论文主方法。

### M6：最后做推理优化

依次增加 grammar/schema constraint、coarse-to-fine head、audio-only verifier、受限 LLM rewrite；VLM 作为额外模式单独报告。每一步都报告收益、额外延迟和失败样例。

## 12. 本轮形成的技术决策

- 保留 **Ledger + unified surface caption** 双层输出；不是只返回若干孤立轨道 caption。
- 下一版数据/Schema 必须补足全局声学环境、track 属性、event 属性和关系；LLM 不得凭空补字段。
- 规则 renderer 作为可复现锚点；LLM 只做受约束 scene proposal，并与经验真实先验做独立消融。
- query 数量由有效数据的 count 分布确定，使用 max-capacity + null；`8/24` 不是已验证常数。
- 当前 CARC 更名理解为 source-removal SFT prototype；在 paired consistency 完成前不把它当论文贡献结果。
- 第一版 query 模型只上最小五项 loss，按课程逐项添加，不一次联合所有设想。
- coarse-to-fine 默认一次 encoder 前向；visual evidence 永远可选，audio-only 是主设置。
- 在 B3-complex-v2 全量 gate 与人工试听通过前，不再启动新的模型方向实验。
