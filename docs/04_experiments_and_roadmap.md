# 实验设计、消融与投稿路线

> 本文给出论文级实验全景。具体工程底座、TAC-style 复现参数和首轮八周执行顺序分别见[实现蓝图](05_base_and_implementation.md)、[复现协议](06_tac_reproduction_protocol.md)和[Pilot 计划](07_pilot_execution_plan.md)。机器可读的 run/ablation 列表见 [`configs/experiment_matrix.yaml`](../configs/experiment_matrix.yaml)。

## 1. 先回答的最小问题

在扩大数据和训练大模型前，用 200 条人工 pilot 和 50k-100k 无标注片段回答：

1. 现有 TAC/MOSS-Audio/Qwen3-Omni 在真实复杂音频上究竟错在哪里？
2. event-set decoder 相比 timestamp-token 自回归基线，是否真的改善重叠与多实例？
3. CARC 相比“相同 pseudo labels 做普通 SFT”是否改善真实域，而非只改善 synthetic mixtures？
4. 改善来自更好定位、减少幻觉，还是单纯变得保守而漏检？
5. 0.1 s grid 是否有可测收益，还是标注/encoder 已限制到 0.25-0.5 s？

任何一个主结论若在 pilot 上不成立，都应修改方法，不要用更大数据掩盖。

## 2. Baselines

### 2.1 必须比较

- **TAC**：最直接的 timestamped caption baseline；按其 brief/detailed、merge/activity/resolution prompt 复现 0.1 s 输出。
- **MOSS-Audio 4B/8B Instruct**：统一 speech/music/sound 和 timestamped ASR 的开放基线。
- **TimeAudio**：时间 marker + absolute time encoding + dense caption；测试其公开 checkpoint/代码。
- **SpotSound**：用于 query grounding、negative event 和 repeated spans 子任务；不能把它当全量 caption baseline。
- **TEMPO**：若作者公开 checkpoint、数据和评测脚本则纳入；否则只报告论文数字，不做不可验证的“复现”。
- **Qwen3-Omni / Audio Flamingo 3**：强通用 LALM；使用固定 prompt 和语法解析。
- **同 backbone 的自回归 baseline**：与 SceneLedger 共享 encoder、LLM、训练数据和参数预算，只移除 event slotter/CARC。这是最重要的公平对照。

### 2.2 专项专家上界

- speech：Whisper/MOSS-Transcribe-Diarize + pyannote/EEND/CSS；
- sfx：FLAM、Detect Any Sound、AudioSet SED；
- lyrics/music：vocal separator + lyric ASR + MIR tools；
- separation：SAM Audio 或同等级 universal separator。

专家 cascade 不满足单模型目标，但能说明错误来自统一建模还是各单项本身就做不到。

### 2.3 公平性

- 所有生成模型限制相同 audio duration、sampling rate、输出 schema 和最大 token；
- prompt 在 dev 冻结，test 不调；
- 同时报告“原生可解析率”和修复格式后的分数；
- 闭源 API 固定日期、版本、temperature 和重试策略；
- 训练数据可能重叠的模型单独标记，不能与严格无泄漏模型混成一个结论。

## 3. 评价指标

### 3.1 Typed Semantic-Temporal F1 (ST-F1)

将预测事件 $p_i$ 与真值 $g_j$ 的匹配权重定义为：

$$
w_{ij}=\mathbf{1}[z_i=z_j]\cdot s_{sem}(c_i,c_j)^\alpha\cdot
\operatorname{tIoU}(\mathcal{S}_i,\mathcal{S}_j)^\beta,
$$

其中 tIoU 对 interval unions 计算，支持 repeated spans。使用 Hungarian matching 最大化总权重 $W$，再计算：

$$
P=W/|P|,\qquad R=W/|G|,\qquad ST\text{-}F1=2PR/(P+R).
$$

必须同时报告 soft precision 和 recall，防止通过少预测来降低 hallucination。`s_sem` 至少使用一个冻结、独立于训练 reward 的 audio-text/text-text model；再用人工 pairwise 评价核验。不能训练和评价都只用 FLAM，也不能只用 LLM judge。

### 3.2 时间指标

- segment F1：100 ms bins；
- event F1：分别使用 onset collar 0.1/0.25/0.5/1.0 s；
- interval mIoU 与 R@IoU 0.3/0.5/0.7；
- matched onset/offset MAE 的 median、p90；
- uncertainty-aware NLL/coverage：预测边界分布是否覆盖人工可接受区间；
- repeated-instance AP：同一 query 的多段定位；
- transient subset：持续时间 <500 ms 的专门结果。

0.1 s 结论主要看 100 ms segment F1、0.1 s collar、boundary MAE 和人工分歧，不看输出字符串是否有一位小数。

### 3.3 Speech 和 lyrics

- speaker-attributed cpWER/tcpWER；
- DER/JER 和 overlapped-speech DER；
- word/line timestamp MAE；
- lyrics WER/CER，按语言和 vocal-to-accompaniment ratio 分层；
- speech-vs-lyrics type confusion；
- 不可听词上的 selective WER：随着 abstention coverage 变化的风险曲线。

### 3.4 Music 和 sfx 文本

- 事件层的 semantic precision/recall；
- instrument/genre/mood/tempo/structure attribute F1 或误差；
- 人工 accuracy、completeness、specificity、hallucination；
- 对同义 caption 使用多 reference 或开放词汇 embedding，不依赖 BLEU/CIDEr 单分数。

### 3.5 可靠性和鲁棒性

- event hallucination rate 与 omission rate；
- ECE/Brier、selective risk-coverage；
- clean-to-corrupted relative retention；
- 对 SNR、T60、echo delay、codec、并发源数、overlap ratio 的性能曲线与曲线下面积；
- counterfactual exactness：加入/移除/平移目标事件后，非目标 ledger 的稳定性和目标 ledger 的正确变化。

## 4. 核心实验表

### Table A：真实统一 caption 主结果

列：ST-F1、semantic P/R、100 ms SegF1、EvtF1@0.25、boundary MAE、hallucination、omission、schema validity、human preference。按 overall 和四个 tag 分解。

### Table B：复杂度/退化曲线

行是模型，列是 clean、2/3/4/5+ sources，SNR 区间，T60 区间，echo、codec。主指标 ST-F1 retention 与 hallucination delta。

### Table C：speech/lyrics 专项

cpWER/tcpWER、overlap DER、lyrics WER、word-time MAE、speech-lyrics confusion、coverage@target error。

### Table D：现有 benchmark 兼容性

TACOS、AudioSet Strong、SpotSound-Bench、AudioCapBench、LibriCSS、DALI/MUSDB subset，以及 TEMPO benchmark（若公开）。这张表证明方法没有为了新 benchmark 过拟合。

## 5. 必做消融

| 消融 | 验证问题 | 预期观察 |
|---|---|---|
| AR timestamp tokens 替代 event slots | set structure 是否必要 | 重叠数增加时 AR 更快退化，重复实例/格式错误更多 |
| 去掉 slot-local evidence，只给全局 audio | 事实性来自何处 | hallucination 上升，语义可能更流畅 |
| 完全去掉 global context | 局部证据是否过强约束 | 事实性保持但场景关系、music 描述下降 |
| 去掉 CARC，保留相同 pseudo SFT | 反事实监督是否超越更多数据 | real robustness 和 removal/insertion consistency 下降 |
| 只做随机 absent negatives | CARC hard negatives 是否必要 | presence accuracy提高但 source-specific removal 泛化较差 |
| 去掉 visual privileged evidence | 视频是否提供有效弱监督 | off-screen 类可能不变，可见同步事件下降；也检查 visual bias 是否反而减少 |
| 去掉 audibility gating | “数学存在”监督是否有害 | 极低 SNR 产生幻觉/过度自信 |
| hard timestamps 替代 soft boundaries | 不确定边界建模价值 | fade/reverb 子集 NLL/coverage 与 MAE 变差 |
| 1.0/0.5/0.25/0.1 s grid | 精细时间的收益-成本 | 确定 0.1 s 是否真实必要，并报告延迟/显存 |
| 单通用 adapter vs modality adapters | 统一训练干扰 | speech/lyrics 与描述性 caption 的 trade-off 变化 |
| 真实 stems vs pseudo-stems | separator 噪声上限 | 给出 CARC 在弱监督下的可信边界 |

## 6. 统计与人工评价

- 所有主指标提供按 clip bootstrap 95% CI；
- 模型比较使用 paired bootstrap/permutation，不只看小数点差异；
- 多随机种子至少覆盖关键同-backbone 实验；
- 人工评价 blind、随机顺序，报告 annotator agreement；
- 提前注册主指标 ST-F1 和两个关键次指标（hallucination、boundary MAE），避免挑指标；
- 公开逐样本预测、解析脚本和错误 taxonomy（在数据许可允许时）。

## 7. 资源策略

### Pilot

- 200 条双标/裁决真实复杂片段；
- 50k-100k 无标注片段；
- 4B-8B backbone 冻结 encoder/LLM，大部分训练 event slotter + LoRA；
- 先在 8 x A100/H100 等级资源做 2-3 个短跑，验证趋势；
- teacher 标注和 SAM Audio 分离往往比主模型 SFT 更耗时，应先缓存并版本化。

### Full study

- 0.5M-1M 真实片段只在 scaling curve 仍有收益时处理；
- benchmark 1,000 条；
- 最少 3 seeds 做关键对照，不要求所有闭源 baseline 多次；
- 训练记录 FLOPs/GPU hours、音频小时数、teacher 推理成本和碳/能耗估计。

算力数字是规划级估计，最终取决于 backbone、音频长度、slot 数和 separation pipeline；不要在论文中把它们写成已完成事实。

## 8. 分阶段里程碑与 go/no-go

### P0：两周 - 数据和评价可用

- 冻结 schema 与 annotation guideline；
- 完成 50 条双标，检查 type agreement 和 boundary variance；
- 跑 3 个开放 baseline，得到真实错误 taxonomy。

**Go**：复杂集相对普通 AAC 有明显 performance gap，标注一致性可接受。  
**No-go**：任务定义导致标注者无法稳定区分 `<speech>`/`<lys>` 或边界；先改 schema。

### P1：四至六周 - 方法最小闭环

- 训练同-backbone AR 与 event-slot 两版；
- 完成真实 stem 上的 insertion/removal/shift；
- 验证 event slots 在 3+ overlap 上的增益。

**Go**：ST-F1 recall 和 hallucination 至少一升一降，且非仅格式收益。  
**No-go**：若 set decoder无益，保留 benchmark，重新聚焦 CARC + structured decoding。

### P2：六至十周 - 无标注真实数据

- 构建 50k-100k pseudo ledgers 和 CARC groups；
- 完成 separator leakage/audibility audit；
- 与相同 pseudo data 的普通 SFT 公平比较。

**Go**：真实 pilot 与至少一个外部 benchmark 有一致收益。  
**No-go**：优先修复 pseudo-stem 和 confidence calibration，不扩数据。

### P3：十二至十八周 - 完整论文

- 完成 1,000 条 WildMix-Cap、隐藏 test 和现有 benchmark；
- 全部消融、鲁棒性曲线、人评与统计；
- 发布模型、schema、评价和合规数据资产。

## 9. 投稿策略

截至 2026-08-08，[ICLR 2027 官方指南](https://iclr.cc/Conferences/2027/AuthorGuidelines) 给出的 abstract/full-paper 截止时间是 2026-09-11/09-16 AOE，只剩约五周。从无实现、无标注 benchmark 的状态强行赶主会，最可能牺牲真实评测和关键消融，不建议作为默认路线。

推荐路线：

1. **主目标：NeurIPS 2027 或 ACM MM 2027**。前者要求方法/学习原则（CARC + evidence ledger）在跨 benchmark 上成立；后者更接受高质量 multimodal data/benchmark 与系统性分析。正式 deadline 尚未发布时不要写死日期。
2. **备选：ACL/EMNLP 2027**。如果贡献更偏结构化生成、语义评价和 speech/music/sound 统一语言接口。
3. **专项备选：ICASSP/Interspeech/ISMIR**。如果最终只在时间定位、多说话人或 lyrics/music 中一项形成强结果，应缩小问题并投稿对应社区，不要用一个弱统一故事覆盖四个不成熟模块。
4. **先公开技术报告/benchmark 不等于抢投**。若担心 2026 新工作快速涌现，可先冻结 task/schema 和 pilot benchmark 的技术报告，再完成主论文；但需考虑目标会议匿名与 concurrent submission 政策。

## 10. Reviewer 可能的质疑与预先回答

### “这不就是 TAC + separator + 更多数据？”

用同 backbone/同数据的 AR vs event-set 对照、CARC vs pseudo-SFT 对照，以及 source-removal exactness 表明贡献来自结构与学习信号，而非组件堆叠。

### “0.1 s 没有意义，人也标不准。”

公开 inter-annotator boundary variance，使用 soft acceptable intervals；分别声明 grid resolution 和 empirical error。对 transient、speech word、fade 三类独立报告。

### “视频教师会制造幻觉。”

视觉永远不能独立成为正标签；加入 visible-but-silent negatives，报告去掉 visual branch 的 precision/recall 变化和 cross-modal hallucination 子集。

### “统一模型不如四个专家，为什么要统一？”

任务价值不要求每个专项都超过专家上界，而要求一次推理输出一致的跨域时间账本，并在相同参数/延迟下胜过通用 LALM。报告专家 cascade 的上界与 4 模型总成本。

### “pseudo-label 只是复制 teacher。”

主监督 CARC 使用可验证的集合变化，不需要 teacher 提供完整真值；在 teacher disagreement、OOD 类和人工 test 上证明 student 的增益，并做不同 teacher 的敏感性分析。
