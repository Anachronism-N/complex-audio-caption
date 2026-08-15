# 显式 Track 锚点、PIT 指标与 Stem 时间证据门禁

更新日期：2026-08-16

## 1. 本轮结论

下一步仍应先完成数据流程与 B3 锚点，不能开始 RL、agent 或更大规模训练。但原来的
“两个 speaker + ambience + 三个 SFX”配方还不能检验 track 建模：每个事件恰好对应
一条 track，模型不需要判断两个不同时段的语音是否来自同一人。

本轮把锚点改为：

```text
speaker-1 utterance A  ───────┐
speaker-1 utterance B  ───────┴─> 同一 speech track
speaker-2 utterance    ─────────> 第二条 speech track
ambience               ─────────> 一条 ambience track
SFX early/middle/late  ─────────> 三条 SFX track

总计：7 个 persisted stems / 7 个 events / 6 个 persistent tracks
```

这使得 event-to-track pointer 第一次成为可学习、可证伪的任务。同时，时间真值不再只
相信 recipe 里的放置参数，而是从渲染后保存的独立 stem 波形重新计算 activity mask。

## 2. 对最新 v6k review 的代码级复核

人工 review 揭示了四类真实问题：caption 与可听内容不一致、部分标注声源被掩蔽、
人声源有噪声、scene-source 组合不合理。这些结论支持停止使用 v6k 做正式训练。

但“mixer 忽略 onset，所有源从 0 s 开始”的解释与代码不符。旧
`scripts/build_real_mix_v6.py` 使用 `start = int(onset * sr)` 插入波形，因此某个
1.2 s 事件不会由该 source 在 0 s 发声。更可能的解释是：

1. 0 s 处是另一 source 的声音；
2. 单源音频包含 caption/类别未覆盖的额外内容；
3. 旧标签记录整段 clip envelope，而非源内部的真实 activity span；
4. 混音后的掩蔽使某个存在于 stem 的事件在人耳上不可辨认。

旧 v6k 音频和 stems 已不在仓库，无法追溯每段听感来自哪个 source。因此该 review
只能作为诊断证据，不能反向修补成可信 ground truth。

`scripts/gen_v6k_heldout_review.py` 也不能作为正式评审工具：它依赖 `/tmp` 中已丢失的
manifest，生成的 audio path 无法重放；缺失 `strict_format_success` 时默认成 True；
所谓 19/20 match 只比较事件数量/类型/四舍五入后的 onset，不比较 caption 语义、
offset 或声源归属。这个比例不能解释为 19/20 正确。

## 3. 新的监督格式

slot-aware target 现在直接输出 track：

```xml
<speech track="T1"><n>0.4</n><slot>turn on the light</slot><n>1.8</n></speech>
<sfx track="T3"><n>1.2</n><slot>a door slams</slot><n>1.6</n></sfx>
<speech track="T1"><n>6.1</n><slot>I am leaving now</slot><n>7.7</n></speech>
<speech track="T2"><n>3.0</n><slot>wait for me</slot><n>4.2</n></speech>
```

这里 T1/T2 只是样本内部的匿名符号，没有跨样本语义。训练 prompt 明确要求：同一持续
声源的多个事件复用一个 track ID，不同说话人或独立声源使用不同 ID。parser 对旧格式
仍保持兼容，但正式报告会记录 `explicit_track_ids_complete`；parser 按类型回填的
legacy track 不得分；inference report 会保留显式 track 是否完整的逐样本状态。

## 4. Persistent track 如何从 stems 得到

`stem` 是一次放置并保存的独立 PCM 波形；`event` 是该 stem 支持的一段带时间文字；
`track` 是同一持续声源在整段 scene 中的身份容器。它们不再假定一一对应。

scene sampler 为 speaker-1 的两句不同 utterance 赋相同的 scene-local
`track_group="speaker-1"`，同时要求两句来自同一 LibriSpeech `source_group`。renderer
仍保存两个独立 stem，便于逐句审计，但把它们合并成一个 track 并保留两个 event。
speaker-2、ambience 和三个 SFX 分别形成独立 track。

这不是用 source 文件名向模型泄漏答案：`source_id`/`track_group` 只保存在 full ledger
和数据审计制品中，训练 target 只看到匿名 T1--T6。

## 5. Pointer 为什么要做 permutation-invariant 评测

reference 的 T1 与 prediction 的 T4 可能代表同一个 speaker。直接字符串相等会把正确
分组判错。新指标先按 type/time 匹配事件，再构造 reference-track × prediction-track
的匹配事件列联表，用 Hungarian assignment 找到最佳一一对齐，最后计算被正确归组的
匹配事件数占 reference event 数的比例。

该指标称为
`permutation_invariant_event_track_accuracy_v1`。未匹配事件、缺失 track 或错误地把
两个 speaker 合并都会失分。例如 reference 分组为 `{e1,e3}` 和 `{e2}`，prediction
把三者都放到同一 track，最高只能正确归组 2/3。

旧 v6k 的 exact-ID pointer=0.34 因 track 名任意而无效。0/100 原始输出含显式 track，
所以新协议将其 PIT-pointer 计为 0；即使 parser 按类型回填后看似形成分组，也不能得到
模型能力分数。

## 6. 时间证据门禁

渲染器为每个 event 在 full ledger 中保留对应 `source_id`，数据验证器据此找到 PCM
stem，并用与监督一致的 frame/hop、RMS threshold 和 merge-gap 重算 activity mask。
对每个 event 检查：

- stem 文件可读取且采样率/声道合法；
- 重算 span 与 ledger span 的时间 IoU ≥ 0.98；
- onset 与 offset 误差各 ≤ 0.1 s；
- 所有 scene 的违反比例必须为 0。

因此“recipe 写了 1.2 s”本身不再是真值证据。只有渲染后的 stem 确实在对应时刻活动，
标签才能授权训练。这个门禁证明的是波形存在和边界一致；混音后是否被掩蔽仍由 active
RMS/可听性门禁及人工 mixture review 判断。

## 7. 代码入口与验收顺序

涉及的主要入口为：

- `sceneledger.data.scene_graph_sampler`：生成同 speaker 的两段 utterance；
- `sceneledger.data.renderer`：保存七个 stems，并聚合成六个 tracks；
- `sceneledger.data.experiment_data`：执行 pointer 可识别性与 stem temporal evidence gate；
- `sceneledger.models.target_formatter`：生成/解析显式 track target；
- `sceneledger.eval.event_matcher`：计算 PIT pointer；
- `sceneledger.cli.infer/evaluate`：冻结显式 track 证据并拒绝旧 report。

必须按下面顺序执行：

```text
source audit（三折、逐类）
  -> 重新生成 120/120/120 specification
  -> 100% rule recipe review
  -> 渲染新 mixture + persisted stems
  -> source identity / complexity / track learnability / temporal evidence gates
  -> 60 条 test 人工试听
  -> frozen test zero-shot
  -> train-only B3 SFT
  -> 同一 test tuned
  -> metrics-v2 + inference-report-v2 + result certification
  -> 盲法语义 A/B review
```

旧 `$SPEC_ROOT` 和 `$EXP_ROOT` 必须废弃，使用新目录。schema、recipe 和 full-ledger
证据已经改变，复用旧 manifest 会被门禁拒绝，也会使新旧结果无法解释。

## 8. 本轮 Go/No-Go

在 GPU 训练前必须同时满足：

1. source audit 和 recipe review 全部通过；
2. train/val/test source groups 互斥；
3. 自动结构门禁确认每个复杂 scene 恰好为七 stem、七 event、六 track；
4. 每个 scene 至少一个 multi-event track，event pointer 100% 完整；
5. stem 重算时间证据零违反，replay/stem-sum 通过；
6. 60 条 test mixture severe failure=0、overall failure≤6；
7. inference report 使用 v2，所有样本都有显式 track 完整性状态；缺失 track 的样本
   pointer 计 0。

任何一项失败都先修数据或协议，不启动 1,000-step B3。锚点通过后才扩到 1k/5k，之后
再单变量比较 recaption、LLM recipe、masking-aware mix、专家模型或 RL。
