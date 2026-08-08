# SceneLedger 实现与实验运行手册

本文把核心研究想法落实到当前仓库的代码接口、数据产物、训练次序和验收门槛。目标不是一次性堆叠
分离、LLM、VLM、RL 和音频编辑器，而是让每一个组件都对应一个可证伪的实验。

## 1. 当前可以直接运行的闭环

```text
授权单源音频
  -> SourceManifest + group-safe split
  -> TAC-mini（受控基线）/ TAC++（复杂声学退化）
  -> SceneLedger + tagged caption + exact stems
  -> MOSS SFT 数据 / MOSS encoder feature cache
  -> Track–Event slot 结构训练与 canonical ledger 解码
  -> 常规 event 指标 + Exact-CARC 反事实指标
  -> structure-aware preference pairs
```

当前 CPU 代码已经覆盖上述接口和单元测试。真正的 MOSS 推理/LoRA、SAM Audio、WhisperX、
pyannote、Demucs 以及大规模训练必须在有 GPU 的服务器上执行。仓库不会伪造这些上游模型的输出。

## 2. 核心表示：语义时间轴与证据时间轴必须分开

一个事件现在有两组时间：

- `event.spans`：语义事件边界。例如一句话真正说出的区间。它是 0.1 s 输出网格上的监督目标。
- `event.evidence.spans`：经过 RIR、echo 或其他声学变换后，波形中仍能测得支持的区间。
- `event.evidence.waveform_uri`：该事件对应的 exact stem 或教师提取波形。

这样做解决了一个容易被忽略的标注冲突：混响尾音可能延续 300 ms，但不能据此把一句话的语义结束
时间延长 300 ms。模型可以学习语义边界，同时利用更宽的 evidence mask 做局部特征池化和幻觉验证。

Canonical JSON 仍是唯一训练/评测真值；XML-like caption 只是模型文本接口：

```xml
<speech track="T1" t="0.7-2.9">...</speech>
<music track="T2" t="0.0-12.8">...</music>
<lys track="T3" t="3.2-6.1">...</lys>
<sfx track="T4" t="4.6-4.9">...</sfx>
```

## 3. 数据构造：先 TAC-mini，再 TAC++，最后拟合真实先验

### 3.1 CPU smoke test

```bash
python -m pip install -e ".[dev,audio,download]"
python scripts/make_toy_sources.py --output data/interim/toy

sceneledger render \
  --sources data/interim/toy/sources.jsonl \
  --config configs/data/tac_smoke.yaml \
  --output data/derived/tac_smoke

sceneledger validate-render data/derived/tac_smoke
sceneledger render \
  --sources data/interim/toy/sources.jsonl \
  --config configs/data/tac_realistic_smoke.yaml \
  --output data/derived/tac_pp_smoke
sceneledger validate-render data/derived/tac_pp_smoke
pytest
```

`validate-render` 会把所有语义 stems 与可选 `T_residual.wav` 相加并重构 mixture。TAC++ 中的压缩、
削波等非线性操作不能分配给某个真实声源，所以差值被显式保存为 residual stem；这既不伪造声源标签，
又保留了 sample-level 可重放性。

### 3.2 TAC++ 复杂声学课程

```bash
sceneledger render \
  --sources /data/sceneledger/manifests/train_sources.jsonl \
  --config configs/data/tac_realistic.yaml \
  --output /data/sceneledger/derived/tac_pp_v1

sceneledger validate-render /data/sceneledger/derived/tac_pp_v1
```

`tac_realistic.yaml` 已支持：

- source-level causal echo；
- 可选 source-level RIR；
- mixture-level white/pink/brown noise 与 SNR 采样；
- device band-pass response；
- dynamic-range compression 与 clipping；
- speech、lyrics、music、sfx、ambience 以及多说话人密集模板；
- 每个随机参数、seed、source path、placement sample、stem path 和 residual path 的 manifest 记录。

这些概率只是启动配置，不应作为论文最终分布。正确做法是从合规的 web pilot 中统计 source count、
共现矩阵、事件时长、speech/music overlap、SNR、T60、echo、设备带宽和平台 codec，再仅使用 train split
拟合 sampler。dev/test 不能反向调整合成分布。

目前尚未实现真实 codec round-trip、ducking、遮挡和音频编辑模型接口；论文实验中必须把它们列为
后续扩展，不能把当前 band-pass/compression proxy 写成真实平台编码器。

## 4. Exact-CARC：把幻觉问题变成可测的反事实问题

### 4.1 构造

```bash
sceneledger carc \
  --backgrounds /data/sceneledger/manifests/web_backgrounds.jsonl \
  --sources /data/sceneledger/manifests/known_sources.jsonl \
  --config configs/data/exact_carc.yaml \
  --output /data/sceneledger/derived/carc_v1
```

每一行 `pairs.jsonl` 同时包含 `before`、`after`、`shifted_after`、target stem、shifted target stem、
SNR、sample-level placement、`delta_event`、`shifted_delta_event` 和 audibility gate。构造满足
`after = before + target`，因此 add/remove 不依赖 separator 的近似结果。

### 4.2 预测文件约定

同一个 pair 必须输出三个 ledger，sample ID 固定为：

```text
{pair_id}:before
{pair_id}:after
{pair_id}:shifted_after
```

然后执行：

```bash
sceneledger evaluate-carc \
  --pairs /data/sceneledger/derived/carc_v1/pairs.jsonl \
  --predictions outputs/s3_carc/predictions.jsonl \
  --output reports/s3_carc_metrics.json
```

指标含义：

- `add_recall`：可听 source 加入后，目标事件被描述的比例；
- `removal_success`：after 中出现且 before 中消失，防止模型对两条音频输出同一故事；
- `pre_intervention_hallucination_rate`：source 尚未加入时已生成其语义的比例；
- `hidden_addition_rate`：低于 audibility gate 的注入仍被模型描述的比例；
- `shift_detection_recall` 与 `shift_equivariance_mae_sec`：事件时间是否随波形平移；
- `background_event_preservation_recall`：干预之外的场景描述是否保持。

CARC 不能替代真实人工测试集。它是可控的训练约束和诊断集，最终必须证明同等数据量下，CARC 相比
pseudo-label SFT 能在真实复杂场景同时提高弱事件 recall 并降低 unsupported-event rate。

## 5. Track–Event student：结构输出与文本输出解耦调试

`TrackEventSlotDecoder` 预测 track presence/type/activity/source embedding 和 event
eventness/type/activity/boundary/track pointer。`slot_set_loss` 使用两级 Hungarian matching、
BCE+Dice activity loss、pointer CE 和 containment loss。

首先缓存 MOSS features：

```bash
python scripts/extract_moss_features.py \
  --upstream-root third_party/MOSS-Audio \
  --model-path weights/MOSS-Audio-4B-Instruct \
  --ledgers /data/sceneledger/derived/tac_pp_v1/ledgers.jsonl \
  --render-manifest /data/sceneledger/derived/tac_pp_v1/render_manifest.jsonl \
  --output /data/sceneledger/features/tac_pp_v1

python scripts/train_slots.py \
  --features /data/sceneledger/features/tac_pp_v1 \
  --validation-features /data/sceneledger/features/tac_pp_dev \
  --config configs/model/track_event_slots.yaml \
  --output outputs/s1_slots \
  --amp
```

训练脚本保存 `last.pt` 与按 validation loss 选择的 `best.pt`，支持 gradient accumulation、
`--resume outputs/s1_slots/last.pt` 和 `--overfit-samples 32`。validation feature 目录应由独立的
group-safe dev manifest 生成，不能从训练 NPZ 随机切分造成 speaker/song 泄漏。
`configs/model/track_event_slots_smoke.yaml` 是 CPU/单卡接线测试配置，不用于正式结果。

张量输出可直接转成 canonical ledger：

```python
from sceneledger.models.decode import decode_slot_arrays

ledger = decode_slot_arrays(
    outputs,
    sample_id="clip_001",
    duration_sec=20.0,
    event_texts={0: "a man says hello", 1: "steady electronic music"},
)
```

解码器会过滤 null slots、映射 event-to-track pointer、把 event activity 限制到对应 track mask，
并在 activity 为空时回退到 onset/offset head。`event_texts` 目前来自外部/后续局部文本解码器；当前仓库
尚未完成 `local_feature -> MOSS evidence-prefix -> event text` 的训练，这是下一项模型代码，而不是已完成能力。

建议服务器上的最低调试顺序：

1. 在 32 条 TAC-mini 样本上关闭 dropout 并过拟合结构头；
2. 检查 source-count、activity、pointer，而不是先看自然语言；
3. 在 500/10k synthetic hold-out 上比较 event-only、track-only、pointer、containment；
4. 接入局部文本 decoder，比较 global-only、time-mask pooling、track-mask pooling、local+gated-global；
5. 最后混入 CARC paired batch，避免多个 bug 同时出现。

## 6. 显式教师不是最终推理级联，而是可审计监督生成器

`sceneledger.teachers` 定义三个插件协议：

- `TrackProposer`：diarization、VAD、Demucs、SAM Audio 或 open-vocabulary SED 提出候选 track；
- `TrackCaptioner`：ASR、lyrics transcription、MOSS/music/sfx captioner 对候选证据生成内容；
- `CaptionVerifier`：target/residual margin、ASR 一致性、audio-text support 或 AV support 做
  accept/correct/abstain。

`TeacherPipeline` 执行 propose -> caption -> verify -> deduplicate -> residual round，并输出带 proposer、
captioner、verifier provenance 的 ledger。建议适配顺序：

1. speech：VAD/pyannote proposal + WhisperX transcription；
2. music/vocal：Demucs proposal + MOSS caption；
3. sfx：SED proposal 或 SAM Audio extraction + MOSS/FLAM caption；
4. rule/audio verifier；
5. 最后才接 LLM 重写和 VLM 辅助。

LLM 只能合并重复、修正语法和按 ledger 重写，不能新增没有 evidence pointer 的事件。VLM 对画外音、
拟音和视觉诱饵会给出错误先验，因此 visual evidence 只能提高/降低置信度或提供匿名 identity，不能越过
audio support gate 强行添加事件。

## 7. DPO/RL：先构造可审计 hard negatives

```bash
sceneledger build-preference \
  --ledgers /data/sceneledger/derived/tac_pp_v1/ledgers.jsonl \
  --render-manifest /data/sceneledger/derived/tac_pp_v1/render_manifest.jsonl \
  --output /data/sceneledger/derived/tac_pp_v1/preferences.jsonl \
  --negatives-per-sample 4
```

当前生成器覆盖 hallucination insertion、event omission、timestamp shift、event type swap、track pointer
swap、event duplication 和 overlong span。输出包含 `chosen`、`rejected`、`negative_type`、audio path 和 seed。

推荐先做 DPO，并逐类人工抽检至少 100 对。只有 reward 在人工偏好集上方向正确、schema validity >99%、
event count 与输出长度稳定时才考虑 GRPO。RL 不是论文成立的必要条件；若不稳定，应保留 hard-negative
评测和 DPO，停止在线 RL。

## 8. 可直接执行的实验矩阵

| ID | 父实验 | 唯一变化 | 必须报告 |
|---|---|---|---|
| B0 | — | MOSS zero-shot canonical prompt | 格式成功率、四类 recall、幻觉 |
| B1 | B0 | TAC-mini static SFT | 普通 SFT 增益 |
| B2 | B1 | TAC-style 动态混音/时间 token/加权 CE | paper-spec baseline |
| B3 | B2 | 同一输出联合 speech/lyrics，无外部 ASR 拼接 | SA-WER、lyrics error、冲突率 |
| S1 | B3 | Track–Event slots + pointer | source-count、overlap recall、pointer |
| S2 | S1 | local evidence + gated global | unsupported event、risk-coverage |
| S3 | S2 | Exact-CARC paired objective | add/remove/shift/hidden/preservation |
| S4 | S3 | audited DPO | 各类 hard-negative 成功率 |

关键消融至少包括：无 track、无 pointer、无 containment、无 local evidence、无 residual verification、
无 audibility gate、CARC 改为等量 pseudo-label SFT、TAC-mini 改为 TAC++、真实先验改为均匀模板。

## 9. 阶段门槛与建议开发顺序

### Gate A：数据层

- 固定 seed 音频 hash 一致；
- `validate-render` 精确重构通过；
- group split 无 speaker/song/video 泄漏；
- 50 条 TAC-mini 与 100 条 TAC++ 人工试听无明显标签错位；
- 合成/真实的 source count、overlap、SNR、T60 等分布有报告。

### Gate B：基线层

- B0/B1/B2 都能输出同一 schema；
- B2 至少在 synthetic temporal 指标上超过 B1，且 real pilot 不退化；
- B3 相比 “TAC caption + 外部 Whisper 拼接” 降低重复和冲突。

### Gate C：结构层

- 32 样本结构过拟合接近完美；
- 打乱 target 顺序结果不变；
- S1 在真实 overlap subset 超过 B3；
- S2 降低 unsupported events，且 speech/lyrics 不明显退化。

### Gate D：反事实层

- S3 提高 add/removal/shift，同时 hidden addition 不上升；
- background preservation 不下降；
- 收益能迁移到人工 WildMix pilot；
- 若只提高 injected-event 指标，则停止扩数据并审计 audibility、source domain 与 separator leakage。

## 10. 本轮实现边界与下一批代码

已完成：TAC++ 基础退化、residual exactness、evidence spans、CARC evaluator、hard-negative builder、
slot-to-ledger decoder、teacher protocols/orchestrator 与 CPU tests。

下一批代码建议严格按以下顺序推进：

1. 实现 `local_feature -> MOSS prefix -> event text`，先在 oracle event masks 上训练；
2. 写 WhisperX/pyannote、Demucs 和 MOSS captioner 的真实 teacher adapters；
3. 增加 teacher audit exporter（accepted/rejected/reason/target/residual waveform）；
4. 从 web train pilot 拟合 scene-prior config；
5. 实现 CARC paired loss 与 batch sampler；
6. 增加 WER/cpWER/tcpWER、lyrics CER 和置信度校准指标；
7. 为大规模 slot 训练增加 variable-length batching/DDP；
8. 在获得许可和算力后再接 SAM Audio、音频编辑模型与 AV verifier。

上述边界应原样写进实验记录，避免把“接口已存在”误写成“上游模型已经跑通”。
