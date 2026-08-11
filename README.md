# Complex Audio Caption / SceneLedger

面向真实复杂声景的统一、细粒度、带时间戳音频描述研究方案（调研冻结日期：2026-08-08）。

## 一句话结论

仅仅把时间 token 加进 Audio LLM 已经不足以构成新贡献。TAC、TimeAudio、SpotSound、MOSS-Audio 和 TEMPO 已分别覆盖了 dense caption、时间定位、统一音频理解或多任务时间戳。更有潜力的方向是：让每条文字描述绑定可定位的声学证据，并通过真实无标注音视频构造“加入、移除、平移、污染某个声源后，事件集合应如何变化”的反事实监督。

本仓库提出暂名 **SceneLedger** 的方案：模型在一次前向推理中，把一个混合音频解析为可重叠的事件集合，并序列化成包含 `<speech>`、`<music>`、`<lys>`、`<sfx>` 的单一 caption。时间使用 0.1 s 网格，但论文将严格区分“0.1 s 输出分辨率”和“真实边界误差达到 0.1 s”。

```xml
<music id="M1" t="0.0-12.8">轻快的电子伴奏持续播放，鼓点逐渐增强。</music>
<speech speaker="S1" t="0.7-2.9">一名男子快速说道：“我们现在开始。”</speech>
<lys singer="V1" t="3.2-6.1">“take me home tonight”</lys>
<sfx id="E1" t="4.6-4.9">近处传来一次玻璃破碎声。</sfx>
<speech speaker="S2" t="4.7-7.0">另一名说话者在音乐和破碎声上方回应。</speech>
```

## 核心贡献候选

1. **Hybrid Track–Event Ledger**：用显式分轨教师产生可审计的 track-level 监督，最终学生在一次前向中以 permutation-invariant track slots 表示持续声源、以 event slots 表示最小 caption 单元，并学习 event-to-track pointer、100 ms activity 与局部声学证据；避免纯 cascade 的分离误差，也避免自回归模型先“讲故事”再猜时间。
2. **Counterfactual Acoustic Remix Consistency (CARC)**：对真实无标注音视频做声源加入、移除、时间平移以及混响/回声/噪声变换，直接约束预测事件集合满足并集、差集、时间等变和可听条件下的不变性。
3. **WildMix-Cap benchmark**：从真实短视频构建人工复核的复杂场景测试集，覆盖多说话人、音乐-人声、lyrics、环境声、混响/回声/噪声和强重叠，提供边界不确定区间、说话人归属及统一标签。
4. **可复现的分解式评价**：分别报告事件语义-时间匹配、边界误差、speaker-attributed WER、歌词错误率、幻觉/漏检、校准和随 SNR/T60/并发声源数变化的鲁棒性曲线，不用单一 BLEU/CIDEr 或单一 LLM judge 掩盖失败。

## 为什么不是已有工作的简单重复

- [TAC](https://arxiv.org/abs/2602.15766) 已能输出 `[music]/[sfx]/[speech]` 与 0.1 s 时间，但语音转录是后接 Whisper，训练以可控合成混音为主；论文也明确承认 sim-to-real gap 和音乐细节不足。
- [TEMPO](https://openreview.net/forum?id=LoXjHBlPEd) 已统一五种 timestamping 任务，但依靠不同任务 prompt；音乐监督主要来自 Slakh2100 合成 MIDI，目标不是同一真实复调场景的一次性统一描述。
- [MOSS-Audio](https://arxiv.org/abs/2606.01802) 已覆盖 speech/sound/music、歌词与时间感知，但没有把“每个生成短语必须由一个局部事件证据支持”作为结构约束，也没有专门验证强混响、回声、噪声和多源重叠下的统一 caption。
- [SpotSound](https://arxiv.org/abs/2604.13023) 通过正/负 query 抑制不存在事件的时间幻觉，但它是 query-based grounding；论文把 polyphonic scenes 和 repeated multi-instance localization 明确列为后续问题。
- [AudioChat](https://arxiv.org/abs/2602.17097) 和 [Audio-Omni](https://arxiv.org/abs/2604.10708) 的重点分别是复杂音频故事的生成/编辑与跨 speech/music/sound 的统一生成编辑，并非本任务的真实复杂声景时间戳 caption。

## 文档导航

- [相关工作与差距矩阵](docs/01_related_work.md)
- [SceneLedger 方法设计](docs/02_sceneledger_idea.md)
- [无标注数据与 WildMix-Cap 基准](docs/03_data_and_benchmark.md)
- [实验、消融、资源与投稿路线](docs/04_experiments_and_roadmap.md)
- [开源底座选择与实现蓝图](docs/05_base_and_implementation.md)
- [TAC-style 基线复现协议](docs/06_tac_reproduction_protocol.md)
- [首轮 Pilot 执行计划](docs/07_pilot_execution_plan.md)
- [Hybrid Track–Event Ledger 详细方法](docs/08_hybrid_track_event_idea.md)
- [数据、音频编辑与 LLM/VLM 验证](docs/09_data_editing_and_verification.md)
- [训练课程、偏好对齐与可验证奖励](docs/10_training_rl_and_rewards.md)
- [逐步开发计划与验收标准](docs/11_development_plan.md)
- [复现指南](docs/12_reproduction_guide.md)
- [Anchor-first：TAG 2021 完整复现协议](docs/13_anchor_first_tag_reproduction.md)
- [机器可读实验矩阵](configs/experiment_matrix.yaml)
- [机器可读复现锚点与门槛](configs/reproduction_anchor.yaml)
- [机器可读开发阶段](configs/pipeline_stages.yaml)
- [规范化事件账本 JSON Schema](schemas/sceneledger.schema.json)
- [Track–Event Ledger v0.2 JSON Schema](schemas/track_event_ledger.schema.json)
- [BibTeX 参考文献](references.bib)

## 当前建议

项目改为 **anchor-first**。第一优先级是完整复现 ICASSP 2021 Text-to-Audio Grounding：固定论文时代代码、标签、特征、训练和官方 evaluator，依次通过数据审计、单 seed 论文指标、随机查询诊断和三 seed 稳定性四个门槛。`runs/tag2021/reproduction_summary.json` 的 `pass=true` 之前，不继续增加新 loss、RL、agent 重写或合成数据变量。

现有 `B1/B2` 和 TAC-mini 结果降级为工程探索结果；schema、parser、evaluator、数据审计和 renderer 继续保留。复现通过后，先在完全相同的数据和 evaluator 上只替换现代 grounding backbone，再逐级扩展到多查询、联合 `<sfx>` 事件、speech/music/lyrics 和复杂声学环境。MOSS-Audio 仍是后续统一模型候选，TAC 仍是最终输出规格参考，但二者不再充当第一阶段的可复现锚点。

截至 2026-08-08，ICLR 2027 的摘要与全文截止日期分别是 2026-09-11 和 2026-09-16；从空仓库开始完成高质量 benchmark、模型与充分消融并不现实。主线更适合瞄准 NeurIPS 2027 或 ACM MM 2027（正式日期发布后再确认），并把 ICLR 2027 作为仅在已有实现和算力成熟时才考虑的高风险窗口。

## 数据合规原则

互联网音视频只能在确认研究使用权、平台条款、隐私和版权边界后进入训练。默认只公开可再分发音频、平台 ID/时间段、派生标注及构建脚本；不要直接公开未经授权的原始 Bilibili、Instagram 或 TikTok 媒体。严格按音频指纹、视频 ID、上传者和音乐作品做 group split，避免同源泄漏。
