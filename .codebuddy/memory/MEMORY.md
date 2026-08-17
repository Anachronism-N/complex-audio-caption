# MEMORY.md — complex-audio-caption (SceneLedger)

> 长期记忆。最近更新：2026-08-17（当日精简整合）。

## 项目定位

**SceneLedger**：音频大模型对复杂混合音频一次前向输出**可重叠、带 0.1s 时间戳的结构化事件集合**（`<speech>/<lys>/<music>/<sfx>`），而非顺序长文本。中心命题：根本困难是**声源归属**——先确定有几个可听声源、各自何时活跃、产生哪些事件，之后才生成文字。

四大创新：① Track(8)–Event(24) 两级排列不变槽位 + event→track pointer（匈牙利匹配训练）；② Evidence-first（事件文字只读自己的局部时频证据 + ≤2 全局 token，时间由边界头确定性产生，不由 LLM 生成）；③ CARC 反事实重混一致性（集合代数监督 + 可听性门控）；④ WildMix-Cap 基准（1000 条真实片段，五层级，边界标最早/最佳/最晚）。

## 技术栈与约定

- 底座 **MOSS-Audio-4B-Instruct** + LoRA(310M 可训)，单卡 40GB，一次训练 8-25 分钟
- 包在 `src/sceneledger/`，schema `0.2.0`，`TIME_RESOLUTION_SEC=0.1`；pydantic 全 `extra="forbid"`，Span 入口即量化到 0.1s 网格
- 渲染 CLI：`python -m sceneledger.cli.render --config configs/data/X.yaml --output-dir ... [--validate]`
- 防泄漏切分 `group_split()` 按"源路径集合 sha1"分组，不按 clip 随机切
- 文档 `docs/NN_*.md`（02=方案主体、03=数据与基准、04=实验路线、13=结果汇总、15/16=合成音频质量、17-22=real_mix v2-v6k review、23=会议对齐、24=数据源盘点、25=下载清单）；`configs/pipeline_stages.yaml` 是机器可读阶段定义

## Git 仓库

- origin = `https://github.com/Anachronism-N/complex-audio-caption.git`
- 本地只有 `main`；远程另有若干 `agent/*` 分支
- 2026-08-17 时点：main 领先 origin/main **7 个本地提交未推送**；`scripts/build_real_mix_v10.py` 有未提交修改

## 数据流水线（两套，未打通）

**A. 包内正式渲染**（可复现、三道 gate：重放 hash 一致 / stems 求和==dry_mixture / ledger 校验）：`configs/data/*.yaml → cli/render.py → scene_graph_sampler → renderer → schema.Ledger → manifests → datamodule`。已产出 tac_mini(500)、b3_unified、b3_5k(5000)、b3_v2。

**B. `scripts/` 真实音源混合**（历史遗留：无文档、硬编码 /tmp）：`download_*.py → extract_sources.py → build_real_mix_v{2..10}.py → convert_v6_manifest.py`。逐源 caption 由 MOSS 实听生成（`label_level="model_prediction"`）。

### 关键路线（2026-08-16）：启用 FileSourcePool，不再写新 build 脚本

- `scene_graph_sampler.py` **早已实现**全部 P0 结构能力：13 个 TemplateID（`overlapping_speakers`、`complex_cocktail`、`lyrics_over_music`/`rich_band`→lys、`repeated_event` 多 span）、完整 `Conditions`、`group_split()`。
- **`FileSourcePool`（:183-205）已实现但从未启用**——所有 config 都是 `pool.kind: synthetic`，file_pool 被注释。需 5 类键：`speech/vocal/music/sfx/ambience`。
- → 正确路线：建 `configs/data/r1_real_pool.yaml`（`kind:file` + 真实路径 + template_weights），一次拿到真实音频+多说话人+lys+多 span+conditions+防泄漏+三道 gate，比再写 build_real_mix_v11.py 便宜一个数量级。
- 环境铁律：素材根目录必须 `data/sources/`，**禁用 /tmp**（含 `HF_HOME`，这是 v6k 丢数据的直接原因）；下载后立即写 `SOURCES.sha256`。

## 核心结论（勿重复踩坑）

- **B0 零样本 F1=0 是预期**：MOSS 能吐时间戳但格式不统一，解析恢复率 0 → 格式纪律需专门训练。
- **隐式 > 显式分离**：mixture 直接 caption 0.948 > oracle stems 0.890 >> Demucs 级联 0.270。**勿做纯级联 pipeline。**
- **排列不变训练是最强证据**：高 overlap(≥0.5) 上 F1 0.714→0.875（+16.1pp）。
- **DETR 式槽位解码器训不起来**（F1 0.08-0.12）；当前 "slot-aware" 只是"输出前加事件计数"的廉价替代，论文不可含糊表述。
- **合成数据 0.948 不代表真实能力**（合成"语音"=正弦波、5000 条音乐同一组和弦），只能定位为格式与时间结构验证。
- **v6 真实 50 条**：F1 0.865 / onset MAE 0.300s / ±0.1s 命中 58.3% / pointer 44.7%；掉分是漏检（变保守）非幻觉。
- **v6k（1000 条）@heldout100 = 0.970 已饱和**：200 条真实+3000 步(0.967) 已超 5000 条合成(0.948)，真实音源路线成立；但扩量后细粒度指标**恶化**（±0.1s 命中 0.842→0.635、onset MAE 0.178→0.262s、pointer 0.34、source_count MAE 1.14）→ **瓶颈在数据结构不在数据量**。
- **v6k 结构诊断（造数根本依据）**：① track↔event 100% 双射且输出格式无 track 字段 → pointer 不可学不可测；② 零多说话人重叠；③ lys=0；④ 全单 span；⑤ GTZAN classical/jazz/blues **34.6% 带人声**（非纯器乐）；⑥ conditions/audibility/boundary_uncertainty 全空；⑦ manifest 不记源路径、eval 按索引切 → 无法 group_split；⑧ speech 仅 2 个 MOSS demo mp3 生成 260 条；⑨ caption 硬截断 200 字符；n_sources∈{2,3}。
- **【事故】v6k 的 1000 条音频已永久丢失**（/tmp 全清）→ F1=0.970 不可复现。`data/derived/real_mix_v6/audio/` 200 条 61MB 是**唯一完好的真实数据**。
- **"0.1s 精度"止损已触发**（docs/02:211）：口径改"100ms 网格+不确定性"，报 ±0.1/0.25/0.5/1.0 四档。
- real_mix v6 是"音源真、场景拼"：零野生录音、零人工标注、时间戳由放置位置直接算。
- 论文 5 个成立条件中，第 3(S2 局部证据)、4(CARC)、5(benchmark) 未达成，恰好是三大创新点。

## 源数据集与扩充

**v6/v7 实际只用了 3 个窄源**：ESC-50（v6 仅 17 类，v7 约 30 类）、GTZAN 3 流派、MOSS demo 2 个 mp3。
**已下载未用存量**：LibriSpeech dev-clean、MUSDB18（150 首分轨，从未使用）、UrbanSound8K（只用了 200/8732）、FMA small（从未真正抽取）。

扩充优先级（详见 docs/24）：
- **P0 MUSDB18-HQ**：vocals→lys（从 0 到有）+ other/drums/bass→纯器乐 music（修 34.6% 污染），一份补两缺口
- **P0 LibriSpeech train-clean-100**：多 speaker + verbatim → 解锁 pointer 与 cpWER
- **P0 TACOS**：唯一能破自我验证闭环的外部真实评测集
- P1：FSD50K（200 类替换 ESC-50）、RIR_NOISES+MUSAN（真实 RIR）、AMI/LibriCSS/AISHELL-4（真实重叠）
- P2：Slakh2100（真分轨+MIDI → CARC）、jamendolyrics（lys 评测金标准）
- 合规：MUSDB18/MedleyDB/MoisesDB 为 CC BY-NC（benchmark 只发 ID+时间段）；GTZAN 无明确许可；CHiME-6/VoxCeleb 需注册勿卡主线

## 待办优先级

**造数原则：事件 F1 已 0.97 饱和，"结构 > 数量"——下一批数据的唯一目标是给 0.1s 边界与声源归属提供监督信号。**

0. 【立即】备份 real_mix_v6/audio；素材重下到持久路径 + sha256；build 脚本加 `--source-root/--out-dir/--n/--seed`，manifest 写 `source_path/source_sha256/speaker_id`。
1. **P0 结构修复版（200-500 条，走 FileSourcePool 路线）**：同源多事件 ≥30%、输出加 track 指针、多说话人 ≥25%、lys ≥5%、多 span ≥15%、n_sources 1-6、填 conditions/audibility/boundary_uncertainty、按源 sha 分组切分（train∩heldout 源=0）、music 人声过滤 <5%、退化 caption 归零、按句截断。**预期 F1 下降属正常且期望**。
2. P1 扩源：接回 LibriSpeech + MUSDB18，ESC-50 补 50 类，真实 RIR 替代合成 RIR。
3. P2 立即启动 200 条真实录音人工标注 pilot（最长前置，裁定 0.1s 是否成立的唯一途径）。
4. 实现 S2 局部证据 + no-local / unrestricted-global 消融。
5. CARC 最小闭环：MUSDB18/Slakh2100 真分轨验证损失 → 5k-10k 真实片段 → 泄漏门槛+人工抽检 → go/no-go 才扩 50k-100k。
6. 工程债：补 real_mix 流水线文档、manifest hash（convert_v6_manifest.py 填的是空串）、v2-v5 归档 scripts/legacy/。

## 会议共识（2026-08-16，docs/23）

- 纪要与仓库方案同构；分歧只在排序：纪要"先扩数据、方法后置" vs 实测证据"结构>数量"。
- 三项增量：① 时间戳作 agentic 生成/精细混音控制信号的反向应用叙事（只作 motivation 不作 contribution）；② **Clue/Claim + 0-1 置信度 + 低置信二次校验**（可解释性唯一可测抓手：ECE/Brier + risk-coverage，应提优先级）；③ RAG 层级语料库降级为教师侧 candidate proposer。
- 场景层只用于规则设定与造数，**不作模型输入**；动态时域变化（走近瀑布等）搁置。
- 转写代号："MOS 4:1"=MOSS-Audio-4B、"TCT"=TAC、"千问3nm"=Qwen3-Omni、"IC 模型"=audio captioner 打标器。
- 时间双轨：轨 A 业务/数据底座 4 周；轨 B 论文主线并行，标注 pilot 立即启动。
- **v6k 的 onset 标签定义本身是错的**（记的是源放置位置，非实际发声时间）→ 人工 pilot 出来前，所有时间指标只有相对比较价值；v7 已改为混音后 RMS 实际活跃区间。
- 待拍板 6 项：一个月里程碑性质；是否承诺 verbatim speech/lyrics；v7 的 F1 下降是否被接受；"0.1s"口径统一；标注 pilot 人力；query/slot 解码器是否继续投算力。

## 下载工程与已就位数据（2026-08-17）

- **star-proxy.oa.com:3128 代理极慢**（KB/s 级）只用于连通性；**hf-mirror.com 直连 6-8 MB/s 为首选**，不挂代理。
- `huggingface-cli download` 在 hub 1.17.0 已改名 `hf download`；大 dataset 会长期阻塞，改用 `wget -c` 循环拉 parquet 分片（`hf-mirror.com/datasets/<repo>/resolve/main/<path>`）+ `xargs -P 4` 并发。
- gated 替代：LibriSpeech 用 `k2-fsa/LibriSpeech`（官方 tar.gz）；MUSDB18-HQ 官方 401 → 用 `danjacobellis/musdb18HQ`(train, WAV parquet) + `roro128/musdb18-hq-flac`(test, FLAC parquet)。
- **已就位**：`/apdcephfs_fsgm3/share_303700817/yikaihuang/dataset/caption/` 下 **LibriSpeech**（dev-clean + train-clean-100，OpenSLR 官方目录结构，40+251 speakers）与 **MUSDB18-HQ 全 150 首**（train 100 + test 50，官方 5-stem wav 布局，44.1kHz stereo，30GB）——可直接接 FileSourcePool。导出脚本与 parquet 中间件保留在同目录。

## 长期警惕

- **自我验证闭环**：伪标签、底座、评价全 MOSS 家族 → 分数好看但无意义，必须引入结构不同的验证信号（ASR 对齐 / FLAM / 重构误差 / 反事实差分）。
- **"变保守"≠"变好"**：精确率与召回率必须同时报告。
- **止损线不要跳过**：主结论若在 pilot 上不成立，修方法，不要用更大数据掩盖（docs/04:15）。

## 投稿

主线 NeurIPS 2027 / ACM MM 2027（ICLR 2027 截止 2026-09 不现实）。
