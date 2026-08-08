# 数据、音频编辑与多模态验证设计

> 本文回答三个工程问题：无标注互联网音视频怎样变成可训练监督；如何比 TAC 更接近真实分布；LLM/VLM 应该怎样参与纠错而不制造新的幻觉。

## 1. 数据策略总览

不要把所有数据都叫作 ground truth。每个样本必须记录来源、可验证程度和允许监督的字段。建议划分五级：

| 级别 | 构造方式 | 精确可知的量 | 不可直接当真值的量 | 默认权重 |
|---|---|---|---|---:|
| A | 原生 stems、人工标注、多轨工程文件 | track、event、时间、文本 | 未标属性 | 1.0 |
| B | TAC++ 程序混音 | 注入来源、变换、时间、track 对应 | 自然场景真实性 | 1.0 |
| C | Exact-CARC：真实背景中加入已知来源 | 被加/删/平移 event 的差分 | 背景中原有全部事件 | 0.8–1.0 |
| D | 真实 mixture 经分离器产生 pseudo tracks | 候选来源及粗时间 | stem 纯度、完整 event 集 | 0.3–0.7 |
| E | 生成式音频编辑模型产生前后对 | 指令语义和预期变化 | 未编辑区域是否真的保持 | 通过验证后 0.2–0.6 |

权重不是单个全局数字，而是字段级 `supervision_mask × confidence`。例如 D 级样本可以高置信监督 `speech transcript`，但低置信监督 `music TF mask`。

## 2. 统一训练样本

数据加载器不直接读取某个数据集专用 JSON，而读取统一的 `TrainingExample`：

```python
@dataclass
class TrainingExample:
    mixture_uri: str
    duration_sec: float
    ledger: dict | None                 # 可为空或不完整
    tracks: list[TrackTarget]
    interventions: list[Intervention]
    condition: AcousticCondition
    supervision: dict[str, Tensor]      # 字段级 mask/weight
    provenance: Provenance

@dataclass
class Intervention:
    op: Literal["add", "remove", "shift", "replace", "degrade"]
    source_id: str | None
    before_uri: str | None
    after_uri: str
    delta_events: list[str]
    params: dict
```

关键规则：

- 每次随机混音都保存完整 renderer manifest；仅保存最终 wav 不可复现；
- 时间统一以原始采样点保存，以 0.1 s 量化仅发生在模型 target builder；
- 文本 seed、LLM 版本、prompt hash、separator/editor checkpoint、阈值均写入 provenance；
- 互联网数据记录平台 ID、上传者 hash、时间段、许可状态和音频指纹，用于删除请求和 group split；
- 同一原视频、翻录音频、同一音乐作品不能跨 train/test。

## 3. Level B：TAC++ 真实分布程序混音

### 3.1 先学场景先验，再渲染

TAC 的核心优势是时间与来源标签精确，短板是人工模板和干净单源可能形成明显 synthetic signature。TAC++ 不直接在真实 web 音频上取伪标签做 SFT，而先从 web 数据估计低维统计分布：

1. 将视频切成 10–40 s 片段，保留镜头边界和非静音区域；
2. 用多个弱教师提取候选：speech VAD/diarization、music probability、vocals probability、开放词汇 sound proposals；
3. 只统计对 teacher 小误差较鲁棒的量：source count 区间、类型共现、并发度、持续时间、相对响度、speech/music overlap、静音比例、场景切换率；
4. 用人工复核的 500–1000 个片段校准这些统计，而不是相信 pseudo caption 文本；
5. 拟合条件场景模型 `p(graph | domain, duration)`，按 vlog、street、gaming、concert、indoor talk 等 domain 分层采样。

一个 scene graph 例子：

```yaml
domain: indoor_vlog
duration_sec: 18.0
tracks:
  - {id: T1, type: speech, identity: speaker_a, role: foreground}
  - {id: T2, type: music, identity: bgm, role: background}
  - {id: T3, type: sfx, identity: dishes, role: intermittent}
relations:
  - {kind: overlap, left: T1, right: T2, ratio: 0.82}
  - {kind: ducking, trigger: T1, target: T2, gain_db: -7}
conditions:
  room: small_reverberant
  capture: phone_far_field
```

LLM 可以把统计约束转成多样 scene graphs，但 graph validator 必须检查来源数、时间范围和分布约束；LLM 不负责最终时间真值。

### 3.2 Renderer 顺序

每条 source 独立执行：

```text
source decode
→ resample/channel normalize
→ time stretch/pitch/formant transform
→ source activity placement
→ source-specific room impulse response / early reflection
→ distance and direction filter
→ gain/ducking/occlusion automation
→ sum sources
→ device response + background/self noise
→ compressor/limiter/clipping
→ AAC/Opus/transcode simulation
```

不能只在最终 mixture 上统一加混响，因为真实场景中各来源距离、方向和直达混响比不同。echo 至少包含：

- acoustic echo：延迟 50–600 ms、衰减和频率响应可变；
- device/playback echo：远端语音或媒体经扬声器再被麦克风采集；
- repeated event：真实重复发声，不能误标成 echo。

### 3.3 两套时间定义

混响尾部会使“事件什么时候结束”没有唯一答案，因此保存：

- `semantic_span`：原始发声/动作有效活动区间，用于 caption 主时间；
- `evidence_span`：加入房间脉冲响应后可检测声学能量区间，用于 evidence mask；
- `boundary_uncertainty`：来自活动阈值、标签源和混响尾部的容差。

论文报告 0.1 s 序列化网格，但边界指标同时按 `±0.1/±0.25/±0.5 s` tolerance 与连续误差报告，不能把网格分辨率描述成真实精度。

### 3.4 防止模型识别“合成痕迹”

- source corpus、RIR、noise、codec 组合按作品/录制设备 group split；
- 训练中混入未经编辑的真实片段，并用 domain adversarial head 检查模型能否轻易识别 synthetic/real；
- 在 hold-out 上训练一个 synthetic detector；若 AUC 很高，说明 renderer 仍有捷径；
- 比较真实与合成的并发度、SNR、spectral flatness、loudness range、T60、codec、event duration 分布；
- 不把所有 sfx 放在整数秒，不让所有 source 都从干净静音开始。

## 4. Level C：Exact-CARC，无需知道真实背景标签

### 4.1 核心构造

取真实无标注背景 `x` 与一个已知单源 `s`，经可记录变换 `A` 后构造：

$$
x^+ = \operatorname{mix}(x, A(s)).
$$

我们未必知道 `x` 的完整 event 集，但精确知道从 `x` 到 `x+` 新增了来源 `s`。因此训练的是预测差分：

$$
F(x^+) \ominus F(x) \approx e_s,
$$

以及非目标事件保持：

$$
F(x^+) \setminus e_s \approx F(x).
$$

同一对样本反向使用就是 exact removal，不需要 separator 生成“移除后音频”；这避免把 separator artifact 当成 removal cue。

### 4.2 四类可验证干预

1. `ADD(s, τ)`：只允许增加一个与 `s` 匹配、时间平移 `τ` 的 event；
2. `REMOVE(s)`：从合成 mixture 回到原始 `x`，该 event 必须消失；
3. `SHIFT(s, Δ)`：语义与 track identity 不变，activity 与时间戳整体平移；
4. `DEGRADE(x, c)`：对整个场景加噪声、混响、codec 或 EQ；可听时 ledger 应基本不变，置信度允许下降。

对低 SNR 注入不能强制模型描述。先用人类校准的 audibility model 或双教师判断 `s` 是否可听：

```python
if audibility < tau_hidden:
    target = "must_not_add"
elif audibility > tau_audible:
    target = "must_add"
else:
    target = "ignore_for_positive_loss"
```

### 4.3 CARC 的匹配损失

先在 `F(x)` 和 `F(x+)` 的非干预事件间做 Hungarian matching，再计算：

- `L_delta_add/remove`：目标 event 的出现/消失；
- `L_shift`：边界随 `Δ` 等变；
- `L_preserve`：非目标 event 的 type、text embedding、track identity 与 activity 保持；
- `L_evidence_delta`：目标 track mask 在干预时间-频率区域增加，其他区域尽量不变；
- `L_confidence`：不可听/模糊样本应降低 confidence 或 abstain。

这比对 `x` 先生成完整 pseudo caption 再 SFT 更稳，因为监督只落在我们真正知道的因果变化上。

## 5. Level D：真实分离 pseudo tracks

参考 Audio-Omni 的真实数据支路，执行：

```text
MLLM/audio model 提出来源类别
→ text/visual/span prompt 送入 SAM Audio
→ 得到 target 与 residual
→ 两侧独立 caption/grounding
→ 重构、残留、相似度、VAD/CLAP/FLAM 过滤
→ 小比例人工抽检并校准置信度
```

SAM Audio 是 promptable separator，不是完整 scene parser；proposal 召回不足会导致漏轨。每个 pseudo track 必须通过：

- `SI-SDR/reconstruction` 或 waveform sum consistency；
- target 对 query 的支持高于 residual，且 margin 足够；
- target 与已有 tracks 不重复；
- 时间活动与 mixture 局部证据一致；
- speech/vocals 使用独立 ASR/VAD 检查；
- 在可选视频中，AV 证据只能加分，不能覆盖 audio 反证。

阈值通过小规模人工集拟合 precision-first operating point。Audio-Omni 报告真实类别样本经过 VAD/CLAP 等过滤后保留比例很低，这支持“宁缺毋滥”的 teacher 数据策略，而不是把所有 separator 输出当标签。

## 6. Level E：生成式音频编辑数据

音频编辑模型适合制造真实质感的 add/remove/style 样本，但它可能同时改变背景、节奏、空间感或说话内容。因此每个编辑对执行三重验证：

1. **Target check**：目标事件是否按指令出现/消失；
2. **Preservation check**：非编辑区域和非目标事件是否保持；
3. **Identity check**：目标真的是指定类型，而非语义近似的替代声音。

推荐先使用 Audio-Omni 式数据流程作为启发，不把尚未稳定开放的 editor 设为 MVP 依赖。编辑失败样本也有价值：可作为 `do_not_learn` 或 verifier hard negatives，但不能直接作为 positive pair。

## 7. LLM、VLM 与音频验证器的职责

### 7.1 三类验证器

| 验证器 | 能可靠检查 | 不应单独决定 |
|---|---|---|
| Audio verifier | 事件是否有局部声学支持、时间、target/residual 差异 | 复杂常识关系 |
| LLM verifier | schema、逻辑冲突、重复事件、文本归一化 | 声音是否真的存在 |
| VLM/AV verifier | 可见声源、口型/动作同步、镜头语境 | 画面外声音、静默物体是否发声 |

VLM 常见错误是“看见狗就补充狗叫”。因此采用 **audio-gated visual correction**：

```python
if audio_support >= high:
    accept_or_refine_with_video()
elif audio_support <= low:
    reject_even_if_visible()
else:
    use_av_sync_to_resolve_or_abstain()
```

VLM 可以修正 `source_id`（哪个人说话）、动作关系或同义描述；不能在没有声学 proposal 的情况下新建 audible event。未来做音视频 caption 时，可以另设 `<visual>` 事件，但不能混入 `<sfx>`。

### 7.2 Accept / Correct / Abstain

验证不是二分类，而是：

- `accept`：事实字段保持；
- `correct`：只能在证据支持的候选集合内修正 type、track pointer、边界或文本；
- `abstain`：保留不确定性，训练时不对该字段施加强监督；
- `reject`：从 pseudo-label 池剔除，可保留为 hard negative。

所有 correction 保存 before/after、证据分数和执行者。若同一个基础模型既生成又验证，会产生循环自洽；至少需要一个结构不同的模型或可计算信号，例如 ASR 对齐、FLAM、重构误差、反事实差分。

### 7.3 训练集与测试集隔离

- LLM/VLM 可辅助训练标注，测试集必须人工复核；
- 测试标注员先听音频后看视频，分别记录 audio-only 与 AV-supported 判断；
- 对模糊来源、遮蔽语音、歌词不可辨识、混响尾部保存 uncertainty，而不是强行共识；
- 任何闭源模型参与测试标注都只作为候选，不作最终裁决。

## 8. 直接实现的模块划分

```text
src/data/
  schema.py                 # ledger/track/intervention dataclasses
  provenance.py             # hash、许可、模型版本
  scene_prior.py            # 从 web 统计拟合条件分布
  scene_graph_sampler.py    # 采样并校验 graph
  renderer.py               # TAC++ 多源渲染
  rir.py                    # source-specific RIR/echo
  exact_carc.py             # add/remove/shift/degrade pairs
  pseudo_track_builder.py   # proposer→SAM Audio→filter
  editor_pairs.py           # 生成式编辑对与 preservation 检查
src/verification/
  audio_support.py
  reconstruction.py
  asr_alignment.py
  av_sync.py
  llm_schema_check.py
  decision.py               # accept/correct/abstain/reject
```

每个数据 builder 都必须通过：固定 seed 可重放、manifest round-trip、边界不越界、source sum/reconstruction、schema 校验、数据泄漏检查以及 20 个可听音频的人工 smoke test。

## 9. 数据阶段的验收标准

开始大规模训练前必须满足：

- 500 个 TAC++ smoke samples 中 100% 可由 manifest 重放，时间误差不超过 1 sample；
- 至少 50 个复杂混音人工检查，类型/时间/source assignment 正确率分别报告；
- synthetic detector AUC 和真实/合成分布差异被记录，并有一轮 renderer 修正；
- 500 个 Exact-CARC 对中，注入事件 audibility gate 的 precision 达到预设目标；
- pseudo-track 池在 200 条人工样本上的 event precision 优先达到 90%，即使 recall 较低；
- 原始平台 ID、版权/许可状态、hash split 与删除索引均可追踪。

只有这些通过后才生成 50k–100k 训练样本。否则扩数据只会放大错误。

## 10. 与开源工作的映射

- TAC 的公开论文规格用于 renderer、atomic timestamps 与 weighted CE：[paper](https://arxiv.org/abs/2602.15766)。
- AudioChat 的 AudioCopilot 展示了 LLM 规划多来源、逐声渲染再混合的数据生成方式：[paper](https://arxiv.org/abs/2602.17097)。
- Audio-Omni 的真实分离与 Scaper 合成双支路用于设计 Level D/B：[paper](https://arxiv.org/abs/2604.10708)。
- SAM Audio 可作为 text/visual/span-prompted separator，但需遵循其权重访问和许可证：[official repository](https://github.com/facebookresearch/sam-audio)。
- SpotSound 可作为正负 query 的 grounding verifier：[official repository](https://github.com/LoieSun/SpotSound)。

