# MEMORY.md — complex-audio-caption (SceneLedger)

> 长期记忆。最近更新：2026-08-14。

## 项目定位

研究项目 **SceneLedger**：让音频大模型对复杂混合音频一次前向输出**可重叠的、带 0.1s 时间戳的结构化事件集合**（`<speech>/<lys>/<music>/<sfx>`），而不是顺序生成一段长文本。

中心命题：复杂音频描述的根本困难是**声源归属**——先确定有几个可听声源、各自何时活跃、产生哪些事件，之后才生成文字。

四大创新：
1. **Track(8)–Event(24) 两级排列不变槽位** + event→track pointer（匈牙利匹配训练）
2. **Evidence-first**：每个事件的文字只能读自己的局部时频证据 + ≤2 个全局 token；时间由边界头产生、确定性序列化，不由 LLM 生成
3. **CARC 反事实重混一致性**：加入/移除/平移/污染 → 集合代数监督 + 可听性门控（`audibility < tau_hidden` 必须不报，`> tau_audible` 必须报，中间不计分）
4. **WildMix-Cap** 基准：1000 条真实复杂片段，五层级，边界标最早/最佳/最晚

## 技术栈与约定

- 底座 **MOSS-Audio-4B-Instruct** + LoRA(310M 可训参数)，单卡 40GB，一次训练 8-25 分钟
- Python 包在 `src/sceneledger/`，schema 版本 `0.2.0`，`TIME_RESOLUTION_SEC = 0.1`
- pydantic 模型全部 `extra="forbid"`；Span/duration 在入口即量化到 0.1s 网格
- 数据渲染 CLI：`python -m sceneledger.cli.render --config configs/data/X.yaml --output-dir ... [--validate]`
- 防泄漏切分用 `group_split()`，按"源路径集合 sha1"分组，不按 clip 随机切
- 文档编号体系：`docs/NN_*.md`，02=方案主体、03=数据与基准、04=实验路线、06=TAC 复现协议、07=pilot 执行计划、08=Track-Event、09=数据编辑与验证、11=开发计划、12=复现指南、13=结果汇总、15/16=合成音频质量问题、17-21=real_mix v2-v6 人工 review
- `configs/pipeline_stages.yaml` 是机器可读的阶段定义（modules / artifacts / gates）

## 数据流水线（两套，未打通）

**A. 包内正式渲染**（可复现、有校验、有文档）：`configs/data/*.yaml → cli/render.py → scene_graph_sampler → renderer → activity → schema.Ledger → manifests → datamodule`。三道 gate：重放 hash 一致 / stems 求和 == dry_mixture / ledger 校验。已产出 tac_mini(500)、b2_no_template、b3_unified、b3_5k(5000)、b3_v2。

**B. `scripts/` 真实音源混合**（无文档、无 argparse、硬编码 `/tmp`）：`download_sources.py + download_music.py → extract_sources.py + setup_esc50.py → build_real_mix_v{2,3,3b,4,5,6}.py → convert_v6_manifest.py`。素材 = ESC-50 / UrbanSound8K / GTZAN / LibriSpeech / FMA。每版 200 条，逐源 caption 由 MOSS 实听生成（`label_level = "model_prediction"`）。

## 核心结论（勿重复踩坑）

- **B0 零样本 F1 = 0 是预期结果**，不是 bug：MOSS 能吐时间戳但格式不统一，解析器恢复率 0。证明格式纪律需专门训练。
- **隐式 > 显式分离**：mixture 直接 caption 0.948 > oracle stems 0.890 >> Demucs 级联 0.270。**不要做纯级联 pipeline。**
- **排列不变训练是目前最强证据**：高 overlap(≥0.5) 上 F1 0.714 → 0.875（+16.1pp）。
- **DETR 式槽位解码器目前训不起来**（F1 0.08-0.12，连 32 样本都过拟合不了）。当前"slot-aware"只是"输出前加一句事件计数"的廉价替代，论文里不可含糊表述。
- **合成数据的 0.948 不代表真实能力**（`docs/16` 自我否定）：合成"语音"是正弦波、5000 条音乐用同一组和弦 `[220, 246.94, 196, 174.61] Hz`，人耳无法辨认。只能定位为"格式与时间结构验证"。
- **v6 真实音源 50 条**：F1 0.865 / P 0.937 / R 0.857 / onset MAE 0.300s / **±0.1s 命中率 58.3%** / **pointer 44.7%** / source_count MAE 1.12。掉分是漏检（变保守）而非幻觉。
- **"0.1 秒精度"的止损条件已触发**（`docs/02:211`），应改称"100ms 输出网格 + 不确定性"，并同时报 ±0.1/0.25/0.5/1.0 四档。
- **论文 5 个成立条件中，第 3(S2 局部证据未实现)、4(CARC 不确定)、5(benchmark 缺失) 未达成**，恰好是三大创新点。
- **real_mix v6 是"音源真、场景拼"**：零野生录音、零人工标注、时间戳由 `onset + len/sr` 直接算。

## 待办优先级

1. 200 条真实录音人工标注 pilot（最长前置，唯一无法靠算力压缩；也是判断 0.1s 是否有意义的唯一参照）
2. 实现 S2 局部证据 + `no-local` / `unrestricted-global` 消融
3. v6 扩到 1000-5000（先修 GTZAN 标签噪声、纯器乐来源、**歌词轨空缺**、caption 单薄）
4. 统一 semantic_span / evidence_span / boundary_uncertainty 三套时间定义
5. CARC 最小闭环：真分轨验证 → 5k-10k 真实片段 → 泄漏抽检 → 对比同量伪标签 SFT → go/no-go
6. 工程债：脚本参数化解除 `/tmp` 依赖、补 real_mix 流水线文档、补齐 manifest hash、v2-v5 归档

## 长期警惕

- **自我验证闭环**：伪标签、底座、评价若都是 MOSS 家族 → 分数好看但无意义。必须引入结构不同的验证信号（ASR 对齐 / FLAM / 重构误差 / 反事实差分）。
- **"变保守"≠"变好"**：精确率与召回率必须同时报告。
- **止损线不要跳过**：任何主结论若在 pilot 上不成立，应修方法，不要用更大数据掩盖（`docs/04:15`）。

## 投稿

ICLR 2027（截止 2026-09-11/09-16）不现实。主线 NeurIPS 2027 / ACM MM 2027。
