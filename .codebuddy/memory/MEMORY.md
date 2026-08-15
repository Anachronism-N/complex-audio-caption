# MEMORY.md — complex-audio-caption (SceneLedger)

> 长期记忆。最近更新：2026-08-15。

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
- **【2026-08-15 更新】真实音频 3000 步后事件 F1 已饱和，但核心指标零进展**：v6(200条)@heldout20 = 0.967；v6k(1000条)@heldout100 = **0.970**（P 0.99/R 0.97，格式 100%，幻觉 3/漏检 9，per-type music .95/sfx .968/speech .952）。**200 条真实 + 3000 步 (0.967) 已超过 5000 条合成 (0.948)** —— 真实音源路线成立。但扩量后细粒度指标**恶化**：±0.1s 命中率 0.842→**0.635**（四档 .635/.688/.778/.900）、onset MAE 0.178→0.262s、pointer 0.30→**0.34**、source_count MAE 0.95→**1.14**。20 条上的 0.842 是小样本波动。→ **加 5 倍数据对 0.1s 精度与声源归属无效，瓶颈在数据结构不在数据量。**
- **【2026-08-15】v6k 数据结构诊断（实测 1000 条 manifest，后续造数的根本依据）**：
  1. `events-per-track` = {1: 2740} —— track↔event **100% 双射**，零"同源多次发声"样本 → pointer 无训练信号；且输出格式 `<n>N</n><slot><type><|t_xxx|>…</type></slot>` **不含 track 字段**，pointer 由评测侧另行推断 → 0.34 既不可学也不是有效测量。
  2. speech track 数 = {0:740, 1:260} —— **零多说话人重叠**（方案第一号难点）。
  3. **lys 事件 = 0** —— 歌词轨完全缺失。
  4. `spans-per-event` = {1: 2740} —— 无重复实例。
  5. **music caption 含人声关键词 196/566 = 34.6%** → GTZAN classical/jazz/blues **并非纯器乐**，`docs/21`"音乐源问题解决"结论是错的；speech 退化 caption 3.5%（"There is no speech or sound…" 被写进 ledger）、sfx 8.8%。
  6. `conditions`/`audibility`/`boundary_uncertainty` **全空** → 分层评测与 CARC 门控做不了。
  7. manifest `sources` **不记源文件路径**；`scripts/eval_heldout.py:15` 是 `manifest[180:]` 按索引切 → 无法 group_split、无法验证泄漏。素材池极小（speech 仅 ~5 个 mp3 生成 260 条）→ F1 0.970 只证明"同素材不同随机组合"泛化。
  8. caption 硬截断 200 字符，句子断在中间；n_sources ∈{2,3} 从不 ≥4；类型失衡 sfx 69.9%/music 20.7%/speech 9.5%；重叠比例均值 0.628（唯一达标项）。
- **【2026-08-15 事故】v6k 的 1000 条音频已永久丢失**：`/tmp/real_audio`、`/tmp/real_mix_v*`、`/tmp/moss_weights` 全被清空，`find -name "rv6k_*.wav"` = 0 → **最好成绩 F1=0.970 不可复现**（音频+源素材皆无、脚本硬编码 /tmp、manifest 无源溯源、`os.listdir` 顺序不保证）。`data/derived/real_mix_v6/audio/` 200 条 61MB 是**唯一完好的真实数据，必须备份**。
- **"0.1 秒精度"的止损条件已触发**（`docs/02:211`），应改称"100ms 输出网格 + 不确定性"，并同时报 ±0.1/0.25/0.5/1.0 四档。
- **论文 5 个成立条件中，第 3(S2 局部证据未实现)、4(CARC 不确定)、5(benchmark 缺失) 未达成**，恰好是三大创新点。
- **real_mix v6 是"音源真、场景拼"**：零野生录音、零人工标注、时间戳由 `onset + len/sr` 直接算。

## 源数据集：实际用的 vs 存量未用（2026-08-15 核实）

实际进入 v6/v6k 的**只有 3 个且都很窄**：ESC-50 **仅 17/50 类**（≈680 条候选 → sfx 1178 + ambience 736）、GTZAN **classical/jazz/blues**（→ music 566）、MOSS demo assets **约 5 个 mp3**（test_en.mp3 / faker_and_chovy.mp3 等 → speech 260）。

下载过但 v6 弃用的**存量**（可立即动用）：**LibriSpeech dev-clean**（270 speakers / 5.4h / 带转录，唯一能支撑多说话人与 verbatim）、**MUSDB18**（150 首带 vocals/accompaniment 分轨，唯一能同时支撑纯器乐+歌词轨+CARC，**从未使用**）、UrbanSound8K（8732 条）、FMA small（8000 首）。
→ v6 为修"音乐带人声"收窄音源池，反而丢掉了最关键的两个源。

## 待办优先级（2026-08-15 修订）

**造数原则：事件 F1 已 0.97 饱和，下一批数据的唯一目标是给"0.1s 边界"与"声源归属"提供监督信号，"结构 > 数量"。**

0. **【立即】** 备份 `real_mix_v6/audio`（61MB）；源素材重下到工作区内持久路径 + 记 sha256；给 build 脚本加 `--source-root/--out-dir/--n/--seed` 且 manifest 写入 `source_path/source_sha256/speaker_id`。不做这条，事故会重演且永远无法防泄漏切分。
1. **P0 v7 结构修复版（200-500 条，不扩量）**：同源多事件（≥30% track 挂 ≥2 event）、输出格式加 `track=` 指针、多说话人（≥25% clip ≥2 speaker）、lys 轨（≥5%）、多 span（≥15%）、n_sources 1-6、填 conditions/audibility/boundary_uncertainty、统一 semantic_span vs evidence_span、按源 sha 分组切分（train∩heldout 源 = 0）、music 人声关键词过滤 <5%、退化 caption 归零、caption 按句截断。**预期 F1 下降属正常且期望**（任务变难），目标是让 pointer 与 0.1s 首次可学可测。
2. **P1 并行扩源**：接回 LibriSpeech + MUSDB18，ESC-50 补到 50 类，加 UrbanSound8K/FMA，真实 RIR 库替代合成 RIR。
3. **P2 立即启动** 200 条真实录音人工标注 pilot（最长前置；边界标最早/最佳/最晚 —— 裁定 0.1s 是否成立的唯一途径）。
4. 实现 S2 局部证据 + `no-local` / `unrestricted-global` 消融。
5. CARC 最小闭环：先 MUSDB18/Slakh2100 真分轨验证损失 → 5k-10k 真实片段 → 泄漏四重门槛+人工抽检 → 对比同量伪标签 SFT → go/no-go 才扩 50k-100k。
6. 工程债：补 real_mix 流水线文档、补齐 manifest hash（`convert_v6_manifest.py:44-47` 填的是空串）、v2-v5 归档到 `scripts/legacy/`。

## 长期警惕

- **自我验证闭环**：伪标签、底座、评价若都是 MOSS 家族 → 分数好看但无意义。必须引入结构不同的验证信号（ASR 对齐 / FLAM / 重构误差 / 反事实差分）。
- **"变保守"≠"变好"**：精确率与召回率必须同时报告。
- **止损线不要跳过**：任何主结论若在 pilot 上不成立，应修方法，不要用更大数据掩盖（`docs/04:15`）。

## 投稿

ICLR 2027（截止 2026-09-11/09-16）不现实。主线 NeurIPS 2027 / ACM MM 2027。
