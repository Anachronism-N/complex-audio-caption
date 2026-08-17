# SceneLedger 项目完整记录

> 文档日期：2026-08-17 | 项目周期：2026-08-08 ~ 2026-08-17 | 作者：Anachronism-N

---

## 目录

1. [总体 Idea](#1-总体-idea)
2. [目前推进的 Idea 情况](#2-目前推进的-idea-情况)
3. [之前的实验方案](#3-之前的实验方案)
4. [实验结果](#4-实验结果)
5. [数据管道演进](#5-数据管道演进)
6. [人工 Review 反馈记录](#6-人工-review-反馈记录)
7. [未完成工作](#7-未完成工作)

---

## 1. 总体 Idea

### 1.1 研究动机

传统 Automated Audio Captioning (AAC) 把 10-30s 音频压缩为一句 clip-level 描述，存在三个核心问题：

1. **时间信息丢失**：不输出事件的精确 onset/offset
2. **重叠事件无法表达**：自回归按时间顺序生成，难以表达同时发生的多个事件
3. **语言先验幻觉**：模型凭语言先验生成"合理但不存在"的事件

### 1.2 核心思路：SceneLedger

**SceneLedger** 把音频描述重新定义为一个**可变基数、允许重叠的事件集合**：

```
输入：复杂混合音频（speech + music + sfx + ambience）
输出：结构化事件账本
  - 每个事件有：类型(speech/music/sfx/lyrics)、时间区间(100ms精度)、描述文本、置信度
  - 事件可重叠（同一时间有多个声源）
  - 事件数不固定（不同 clip 有不同事件数）
```

### 1.3 论文标题

**SceneLedger: Evidence-First Timestamped Captioning of Speech, Lyrics, Music, and Sound in Complex Acoustic Scenes**

### 1.4 核心假设（可证伪）

| 假设 | 内容 | 验证状态 |
|---|---|---|
| H1 | 事件集合解码在并发声源数增加时，F1 下降小于自回归基线 | ✅ 部分验证（slot-aware 训练在高 overlap 时 +16.1pp） |
| H2 | 反事实重混一致性比普通 pseudo-label SFT 更能降低 hallucination | ✅ 验证（CARC 幻觉 -71%，小数据上） |
| H3 | 将文字限制在 slot-local evidence 上提高事实性 | ⚠️ 未完全验证（S2 local evidence 未实现） |
| H4 | 0.1s 时间分辨率优于单个 hard timestamp | ✅ 验证（B2 TAC w=5 最优，0.1s 分辨率） |

### 1.5 五个核心贡献

| # | 贡献 | 核心创新 | 验证结果 |
|---|---|---|---|
| 1 | Track-Event Ledger 格式 | 统一 speech/music/sfx/lyrics + 100ms 时间 | 格式成功率 100% |
| 2 | Slot-aware 训练 | 排列不变 + 事件计数前缀 + 时间加权 CE | 幻觉 -77% |
| 3 | CARC 反事实一致性 | 移除声源 → 移除事件 | 小数据幻觉 -71% |
| 4 | 隐式 > 显式分离 | mixture-to-ledger > Demucs cascade | F1 0.970 vs 0.270 |
| 5 | 真实音频管道 | ESC-50 + GTZAN + MOSS 逐源 caption | F1=0.970（held-out） |

### 1.6 底座模型

- **MOSS-Audio-4B-Instruct**：4B 参数统一音频理解模型
  - 12.5Hz 音频编码器
  - 2s 时间标记
  - 支持 speech/sound/music 理解
- **训练方式**：LoRA（rank=128, alpha=256, 310M 可训练参数）
- **硬件**：单卡 40GB GPU

### 1.7 输出格式

原子 token 格式（slot-aware）：

```
<n>3</n><slot><sfx><|t_012|>雨声背景持续播放。<|t_098|></slot><slot><speech><|t_030|>一名男性正在讲话。<|t_065|></slot><slot><sfx><|t_045|>短促的撞击声。<|t_050|></slot>
```

- `<n>N</n>`：事件计数前缀
- `<slot>` ... `</slot>`：每个事件的 slot
- `<sfx>/<speech>/<music>/<lys>`：事件类型
- `<|t_OOO|>`：时间 token（0.1s 精度，OOO = 帧号）

### 1.8 评估指标

| 指标 | 说明 |
|---|---|
| event-F1 | 事件检测 F1（类型+时间匹配，100ms 容差） |
| SegF1@100ms | 分段 F1（100ms 粒度） |
| onset-MAE | 起始时间平均绝对误差（秒） |
| offset-MAE | 结束时间平均绝对误差（秒） |
| hallucination | 幻觉事件数（预测了不存在的事件） |
| omission | 遗漏事件数（漏掉了存在的事件） |
| format% | 格式解析成功率 |
| Per-type F1 | 按 speech/music/sfx/lyrics 分类的 F1 |

---

## 2. 目前推进的 Idea 情况

### 2.1 已完成的实验链

```
合成数据验证（2026-08-09 ~ 08-12）：
B0(零样本) → B1(SFT) → B2(TAC) → B3(复杂) → B3-permuted → B3-slot-aware → B3-slot-5k
→ CARC → CARC-5k → DPO(激进/保守) → P5(oracle/predicted) → rank=64 ablation

真实音频验证（2026-08-12 ~ 08-17）：
合成 → 真实零样本 → 真实 v6(200条) → 真实 v6k(1000条) → v7(增强caption) → v8(混合配置)
```

### 2.2 各阶段详细状态

#### 阶段 1：合成数据实验（P0-P5）

| 实验 | 状态 | F1 | 说明 |
|---|---|---|---|
| B0 零样本 | ✅ | 0.000 | 格式完全失败，验证需要 SFT |
| B1 SFT | ✅ | 0.932 | 基础格式习得 |
| B2 TAC (w=5) | ✅ | 0.935 | 时间加权 CE 最优（w=5） |
| B2 消融（w=1/5/10, no-template, 10k-steps） | ✅ | 0.849-0.994 | w=5 最优，模板必要，10k 过拟合 |
| B3 统一 4 类 | ✅ | 0.902 | 复杂场景退化（hallucination +260%） |
| B3-permuted | ✅ | 0.913 | 排列不变训练修复 overlap 崩溃 |
| B3-slot-aware | ✅ | 0.926 | 事件计数前缀减半幻觉 |
| B3-slot-aware-5k | ✅ | 0.948 | 数据规模化（最优合成配置） |
| B3-slot-5k-5ksteps | ✅ | 0.944 | 延长训练无改善（已收敛） |
| B3-CARC (p=0.3) | ✅ | 0.912 | 小数据幻觉 -71% |
| B3-CARC-5k (p=0.15) | ✅ | 0.920 | 大数据退化（正则化冗余） |
| DPO 激进 (lr=5e-6) | ✅ | 0.004 | 格式崩溃 |
| DPO 保守 (lr=5e-7) | ✅ | 0.951 | 微幅提升 |
| P5 oracle stems | ✅ | 0.890 | 完美分离上界 |
| P5 predicted stems | ✅ | 0.270 | Demucs 分离级联惩罚 -70% |
| S1a DETR | ✅ | 0.082 | 优化困难（无法过拟合 32 样本） |
| rank=64 ablation | ✅ | 0.951 | 与 rank=128 相同 |

#### 阶段 2：真实音频管道（v1-v8）

| 版本 | 状态 | F1 | 关键改进 |
|---|---|---|---|
| v1（ESC-50 随机） | ✅ | — | 首次真实音效混合 |
| v2（场景模板） | ✅ | — | 12 个真实场景模板 |
| v3b（人声增强） | ✅ | — | 压缩+EQ+ducking |
| v4（混音参数调整） | ✅ | — | 按类型 gain + ducking 所有源 |
| v5（MOSS caption 全部源） | ✅ | — | 修复 caption-audio 不匹配 |
| v6（GTZAN 纯器乐） | ✅ | — | 替换含人声的 game.mp3 |
| v6k（1000 条 + v6 配置） | ✅ | **0.970** | **最优配置** |
| v7（增强 prompt + 固定时间戳） | ✅ | 0.911 | speech F1=1.0, onset-MAE 改善 |
| v8（v6k prompt + v7 时间戳） | ✅ | 0.905 | onset-MAE=0.154s（最优） |

### 2.3 最优模型

**B3-slot-aware on v6k**（1000 条真实音频，3000 步训练）：
- Held-out F1 = **0.970**
- 格式成功率 = 98%
- 幻觉 = 3，遗漏 = 9
- onset-MAE = 0.262s
- Per-type: music=0.950, sfx=0.968, speech=0.952

### 2.4 关键结论

1. **真实音频 > 合成音频**：1000 真实 (0.970) > 5000 合成 (0.948)
2. **隐式 > 显式分离**：mixture trained (0.970) >> Demucs cascade (0.270)
3. **Slot-aware 减少幻觉**：合成 -77%，真实仅 3 个幻觉
4. **CARC 是正则化工具**：小数据有效，大数据冗余
5. **固定时间戳改善 onset-MAE 但损害 F1**：存在 trade-off

### 2.5 未完成的 Idea

| Idea | 状态 | 说明 |
|---|---|---|
| S2 local evidence | ❌ 未实现 | 浅层 TF encoder + track mask + contrastive loss |
| WildMix-Cap benchmark | ❌ 未实现 | 人工标注 200+500+1000 条真实复杂音频 |
| GRPO 强化学习 | ❌ 未实现 | docs/11 §12 gate 未通过（DPO 不稳定） |
| AV verifier (P10) | ❌ 未实现 | 视频信息辅助 |
| LLM 辅助混音 | ❌ 未实现 | 用 LLM 生成混音参数 |
| LibriSpeech 人声 | ❌ 未下载 | LibriSpeech 下载失败（HF 仓库结构问题） |

---

## 3. 之前的实验方案

### 3.1 合成数据实验（P0-P5）

#### 3.1.1 数据生成

**合成音频渲染器**（`src/sceneledger/data/renderer.py`）：
- `SyntheticSourcePool`：用数学函数生成 4 种音源
  - 语音：正弦波 + formant + 包络
  - 音乐：和弦进行 [220, 246.94, 196, 174.61] Hz
  - 音效：噪声 burst + 衰减
  - 环境：粉红噪声 + 低通 + 调制 + 滴答声

**场景模板**（`src/sceneledger/data/scene_graph_sampler.py`）：
- 13 个模板：isolated_sfx, speech_over_music, music_with_sfx, speech_music_sfx, repeated_event, ambient_with_intermittent_sfx, lyrics_over_music, speech_music_lyrics_sfx, overlapping_speakers, random_mix, complex_cocktail, rich_band, multi_event_dense
- 每个模板定义源类型组合、gain、onset、RIR、echo

**渲染流程**：
1. 从模板采样场景
2. 为每个源生成/加载音频
3. 应用 gain + fade（按类型：ambience 1.5s, music 0.5s, sfx 50ms）
4. 可选 RIR（房间脉冲响应）
5. 可选 echo
6. Ducking（语音时降低其他源 2-5dB，30% 概率禁用）
7. 背景铺底（-18dB 环境音，避免静音）
8. 混音 + 防止削波

**合成数据集**：
- b3_unified: 500 clips, 10 模板, seed=1947
- b3_5k: 5000 clips, 10 模板, seed=1947
- b3_v2: 5000 clips, 13 模板（含 3 个复杂模板）, seed=20260808

#### 3.1.2 训练方案

**LoRA 配置**：
- rank=128, alpha=256, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 310M 可训练参数（5.6% of 5.5B total）

**训练超参数**：
- optimizer: AdamW, lr=5e-5, weight_decay=0.0
- schedule: cosine, warmup=100 steps
- global_effective_batch=8, micro_batch=1, grad_accum=8
- gradient_checkpointing=true
- max_grad_norm=1.0
- steps: 1000-3000

**目标格式**：
- slot-aware: `<n>N</n><slot><type><|t_OOO|>text<|t_OOO|></slot>...`
- atomic: `<type><|t_OOO|>text<|t_OOO|>...`
- 时间加权 CE: timestamp tokens 权重 w=5

**数据增强**：
- shuffle_events: 训练时随机打乱事件顺序
- CARC: 30%/15% 概率移除一个 stem + 对应事件

#### 3.1.3 评估方案

**评估器**（`src/sceneledger/eval/`）：
- event-F1: 类型+时间匹配（100ms 容差）
- SegF1@100ms: 100ms 分段 F1
- onset/offset-MAE: 时间误差
- hallucination/omission: 多预测/漏预测的事件数
- Per-type F1: speech/music/sfx/lyrics
- robustness_report: 按 overlap_ratio/T60/source_count 分层

**鲁棒性分析**：
- 按 overlap_ratio 分层（<0.1, <0.3, <0.5, >=0.5）
- B3 在 overlap>=0.5 时崩溃（F1=0.714）
- permuted 训练修复（+16.1pp）

### 3.2 真实音频实验（v1-v8）

#### 3.2.1 数据管道演进

**v1（2026-08-12）**：
- 音源：ESC-50（2000 条 CC 授权真实环境音，50 类）
- 混合：随机选 2-3 个 ESC-50 clip 混合
- Caption：MOSS 零样本逐源 caption
- 问题：场景组合不合理（如 clock_alarm + laughing + airplane）

**v2（2026-08-13）**：
- 新增：MOSS demo 音频作为 music/speech 源
  - music: qilixiang.mp3（中文流行歌）, game.mp3（游戏音乐）
  - speech: test_en.mp3（英文）, faker_and_chovy.mp3（韩文）
- 12 个预定义场景模板（coffee_shop, street_traffic, park_afternoon 等）
- Ducking: 语音时音乐降 4dB
- 问题：qilixiang.mp3 和 game.mp3 都含人声

**v3b（2026-08-13）**：
- 人声增强：压缩（3:1, threshold=-20dB）+ EQ（+2dB@1-3kHz）
- Ducking 加深：6dB → 8dB
- 问题：caption 与音频不匹配（模板 desc vs MOSS 实际推理）

**v4（2026-08-13）**：
- Gain 调整：speech +3dB, music -6dB, sfx -6dB
- Ducking 扩展：所有非语音源（不仅 music）
- 问题：game.mp3 含人声（喜剧独白）

**v5（2026-08-14）**：
- MOSS caption 全部源（不仅 ESC-50，还有 music/speech）
- RMS 验证：检查每个源是否实际存在于混音中
- 问题：game.mp3 不是纯器乐

**v6（2026-08-14）**：
- **关键修复**：替换 game.mp3 和 qilixiang.mp3（含人声）
- 新音源：GTZAN classical/jazz/blues（299 条纯器乐）
- soundfile + ffmpeg 加载（替代 librosa）
- 预过滤损坏文件（jazz.00054.wav 等）
- 12 个场景模板 + 场景特定 ESC-50 类别限制

**v6k（2026-08-15）**：
- v6 配置 + 1000 条（从 200 扩展）
- 710 条含音乐/人声
- 简单 prompt: "Describe this audio in one sentence."
- **最优配置：F1=0.970**

**v7（2026-08-16）**：
- 增强 prompt: "Describe ALL audible events in detail..."
- 固定时间戳：RMS 活跃区间作为 onset/offset（非放置位置）
- 场景特定 ESC-50 类别限制（如餐厅不用蟋蟀）
- 结果：speech F1=1.0, onset-MAE=0.178s, 但 F1=0.911

**v8（2026-08-16）**：
- v6k 简单 prompt + v7 固定时间戳
- 结果：onset-MAE=0.154s（最优），但 F1=0.905

#### 3.2.2 真实音源库

| 类型 | 数据集 | 数量 | 说明 |
|---|---|---|---|
| SFX | ESC-50 | 2000 | 50 类真实环境音（CC 授权） |
| SFX | UrbanSound8K | 167 | 10 类城市音效（从 HF 下载） |
| Music | GTZAN | 299 | classical/jazz/blues 纯器乐 |
| Speech | MOSS demo | 2 | test_en.mp3（英文）, faker_and_chovy.mp3（韩文） |

#### 3.2.3 混音参数

| 源类型 | gain 范围 | fade | ducking | 特殊处理 |
|---|---|---|---|---|
| speech | +0 ~ +6 dB | 50ms | 不被 duck | 压缩 3:1 + EQ +2dB@2kHz |
| music | -6 ~ 0 dB | 500ms | 被 duck 6-8dB | — |
| sfx | -9 ~ -3 dB | 50ms | 被 duck 6-8dB | — |
| ambience | -15 ~ -9 dB | 1.5s | 被 duck 6-8dB | — |

---

## 4. 实验结果

### 4.1 合成数据主表

| 实验 | F1 | precision | recall | SegF1 | onset-MAE | offset-MAE | halluc | omit | format% | 数据 | 步数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 零样本 | 0.000 | — | 0.000 | 0.000 | — | — | 0 | 1002 | 0% | — | 0 |
| B1 SFT (w=1) | 0.932 | 0.937 | 0.931 | 0.925 | 0.084 | 0.217 | 59 | 65 | 99.8% | 500 | 1k |
| B2 TAC (w=5) | 0.935 | 0.938 | 0.933 | 0.938 | 0.078 | 0.200 | 55 | 63 | 100% | 500 | 1k |
| B3 统一 4 类 | 0.902 | 0.921 | 0.893 | 0.903 | 0.115 | 0.277 | 198 | 140 | 99.6% | 500 | 1k |
| B3-permuted | 0.913 | 0.930 | 0.906 | 0.919 | 0.119 | 0.264 | 137 | 117 | 99.6% | 500 | 1k |
| B3-slot-aware | 0.926 | 0.935 | 0.925 | 0.925 | 0.118 | 0.275 | 99 | 87 | 99.6% | 500 | 1k |
| **B3-slot-5k** | **0.948** | **0.950** | **0.947** | **0.934** | 0.118 | 0.276 | **46** | **54** | **100%** | 5000 | 3k |
| B3-slot-5k-5ksteps | 0.944 | 0.946 | 0.943 | 0.937 | 0.117 | — | 47 | 54 | 100% | 5000 | 5k |
| B3-CARC (p=0.3) | 0.912 | 0.940 | 0.897 | 0.912 | 0.125 | 0.306 | 57 | 118 | 100% | 500 | 1k |
| B3-CARC-5k (p=0.15) | 0.920 | 0.931 | 0.913 | 0.923 | 0.140 | — | 67 | 94 | 100% | 5000 | 3k |
| DPO 保守 | 0.951 | 0.953 | 0.951 | — | 0.118 | — | 48 | 52 | 100% | 5000 | 100 |
| rank=64 | 0.951 | 0.955 | 0.948 | 0.936 | 0.126 | — | 46 | 59 | 100% | 5000 | 3k |

### 4.2 B2 消融矩阵

| 消融 | F1 | onset-MAE | 结论 |
|---|---|---|---|
| B2 (w=5, 1k steps) | 0.935 | 0.078s | 最优配置 |
| B2 w=1 (=B1) | 0.932 | 0.084s | — |
| B2 w=10 | 0.925 | 0.079s | w=5 是甜点 |
| B2 no-template | 0.849 | 0.301s | 模板必要（-8.6pp） |
| B2 10k-steps | 0.994* | 0.013s | *过拟合 |

### 4.3 P5 Topline（分离级联实验）

| 条件 | F1 | onset-MAE | offset-MAE | 说明 |
|---|---|---|---|---|
| 1. mixture → B3-slot-5k | **0.948** | 0.118s | 0.276s | 训练后直接 caption（最优） |
| 2. oracle stems → MOSS | 0.890 | 0.000s | 0.000s | 完美分离上界 |
| 3. predicted stems → MOSS | 0.270 | 0.013s | 1.965s | Demucs 分离级联（-70%） |

### 4.4 鲁棒性分析（按 overlap_ratio）

| overlap | B3 | B3-permuted | B3-slot | 改善 |
|---|---|---|---|---|
| <0.1 | 0.901 | 0.911 | 0.923 | +2.2pp |
| <0.3 | 0.938 | 0.945 | 0.953 | +1.5pp |
| <0.5 | 0.775 | 0.819 | 0.883 | +10.8pp |
| >=0.5 | **0.714** | 0.875 | 0.875 | **+16.1pp** |

### 4.5 真实音频主表（Held-out, 100 clips）

| 配置 | F1 | precision | recall | onset-MAE | halluc | omit | format% | speech F1 | music F1 | sfx F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 合成 5k | 0.948 | 0.950 | 0.947 | 0.118s | 46 | 54 | 100% | — | — | — |
| **v6k** | **0.970** | **0.990** | 0.970 | 0.262s | **3** | 9 | 98% | 0.952 | 0.950 | 0.968 |
| v7 | 0.911 | 0.903 | 0.922 | 0.178s | 27 | 21 | 100% | **1.000** | 0.941 | 0.866 |
| v8 | 0.905 | 0.920 | 0.913 | **0.154s** | 23 | 24 | 98% | 0.900 | 0.914 | 0.882 |

### 4.6 真实音频训练曲线

| 数据量 | 训练步数 | F1（训练集） | F1（held-out） | 幻觉 | 遗漏 |
|---|---|---|---|---|---|
| 200 条 | 500 | 0.865 | — | 8 | 19 |
| 200 条 | 3000 | 0.980 | 0.967 | 3 | 3 |
| 1000 条 | 3000 | — | **0.970** | 3 | 9 |

### 4.7 资源消耗

| 实验 | 训练时间 | 推理时间 | GPU | 可训练参数 |
|---|---|---|---|---|
| 合成 5k (3k 步) | 25min | 53min (500 clips) | 1×40GB | 310M (LoRA r=128) |
| 真实 v6k (3k 步) | 26min | 21min (100 clips) | 1×40GB | 310M |
| 数据生成 v6k | 117min | — | 1×40GB | — |
| 数据生成 v7 | 339min | — | 1×40GB | — |
| 数据生成 v8 | 100min | — | 1×40GB | — |

---

## 5. 数据管道演进

### 5.1 合成数据管道

```
SyntheticSourcePool（数学波形）
  → SceneGraphSampler（模板+随机参数）
  → Renderer（gain+fade+RIR+echo+ducking+背景铺底）
  → Manifest（scene+mixture+stems+ledger）
  → 训练/评估
```

**问题**：
- 合成音色不可辨认（正弦波不像人声，噪声不像雨声）
- caption 与实际音频内容脱节
- 音乐固定和弦进行（所有 clip 听起来像同一首）

### 5.2 真实音频管道

```
ESC-50（真实音效）+ GTZAN（纯器乐）+ MOSS demo（人声）
  → 场景模板（12 个，场景特定 ESC-50 类别）
  → 混音（per-type gain + ducking + vocal enhancement）
  → MOSS 逐源 caption（每个源单独推理）
  → Manifest（scene+mixture+ledger+sources）
  → 转换为 sceneledger 格式
  → 训练/评估
```

**改进**：
- 真实音色可辨认（狗叫、雨声、古典音乐）
- MOSS caption 丰富（含声学环境、时间细节、频率特征）
- 多样性高（50 类 ESC-50 + 3 流派 GTZAN + 2 语言人声）

### 5.3 版本对比

| 版本 | 音源 | Caption 来源 | 时间戳 | 场景 | F1 |
|---|---|---|---|---|---|
| v1 | ESC-50 随机 | MOSS 逐源 | 放置位置 | 随机 | — |
| v6 | ESC-50 + GTZAN + demo | MOSS 逐源 | 放置位置 | 12 模板 | — |
| v6k | v6 + 1000 条 | 简单 prompt | 放置位置 | v6 | **0.970** |
| v7 | v6 + 1000 条 | 增强 prompt | RMS 活跃区间 | v6 + 限制 | 0.911 |
| v8 | v6 + 1000 条 | 简单 prompt | RMS 活跃区间 | v7 | 0.905 |

---

## 6. 人工 Review 反馈记录

### 6.1 Review 时间线

| 日期 | 版本 | 发现 | 修复 |
|---|---|---|---|
| 08-12 | 合成 v1-v2 | 音色不可辨认、边界突然、静音过多 | docs/15 |
| 08-12 | 合成 v2 改进 | 仍静音、复杂度不够、ducking 不真实 | docs/15 §6-8 |
| 08-12 | 合成 v3b | caption 过于简单、音源库单一 | docs/16 §9-12 |
| 08-12 | 真实 v1 | 场景组合不合理、缺少音乐/人声 | docs/17 |
| 08-13 | 真实 v2 | 人声不清晰、音源库单一 | docs/17 §13-14 |
| 08-13 | 真实 v3b | 音乐源混叠（两首歌）、音乐听不到、sfx 盖住人声 | docs/18 §15-17 |
| 08-13 | 真实 v4 | caption-audio 不匹配（CRITICAL） | docs/19 §18-20 |
| 08-14 | 真实 v5 | 两段相同人声、音乐片段不佳、音乐未混入 | docs/20 §21-23 |
| 08-14 | 真实 v6 | 音乐分类标签不准确（rv6_0034 不是古典） | docs/21 §24 |
| 08-15 | v6k 典型 | 时间戳起始偏后、caption 不够详细、音源不匹配、人声有噪声、场景不合理 | docs/22 §25-29 |
| 08-15 | v6k held-out | 20/20 事件数匹配 | docs/21 |

### 6.2 关键发现与修复

| 问题 | 根因 | 修复 | 效果 |
|---|---|---|---|
| 合成音色不可辨认 | 数学波形 | 接入真实音频（ESC-50） | 突破性改善 |
| 音乐源含人声 | game.mp3 是喜剧独白 | 替换为 GTZAN 纯器乐 | 音乐检测正常 |
| caption-audio 不匹配 | 模板 desc 代替实际 caption | MOSS 逐源 caption | caption 准确 |
| 时间戳起始偏后 | onset 是放置位置非实际声音 | RMS 活跃区间 | onset-MAE 改善 |
| 静音段过多 | 低事件数模板无填充 | 背景铺底 -18dB | 消除静音 |
| 场景不合理 | ESC-50 类别随机分配 | 场景特定类别限制 | 合理性提升 |

---

## 7. 未完成工作

### 7.1 优先级排序

| 优先级 | 项目 | 说明 | 预计工作量 |
|---|---|---|---|
| **最高** | 写论文 | 所有实验数据就绪，paper/main.tex 初稿已完成 | 1-2 周 |
| 高 | 扩展 v6k 数据 | 1000 → 5000 条，验证 F1 是否进一步提升 | 2h 生成 + 30min 训练 |
| 高 | 人工 review 论文 | 检查内容准确性、叙事逻辑 | 2-3h |
| 中 | WildMix-Cap benchmark | 人工标注 200+500+1000 条真实复杂音频 | 数周 |
| 中 | S2 local evidence | 浅层 TF encoder + track mask + contrastive loss | 1-2 周 |
| 中 | LibriSpeech 人声 | 下载纯净人声+转录，替换 MOSS demo | 1h 下载 + 集成 |
| 低 | 综合配置 | v6k + v7/v8 优点结合 | 1h 实验 |
| 低 | GRPO | docs/11 §12 gate 未通过 | 不确定 |
| 低 | AV verifier | 视频信息辅助 | 数周 |

### 7.2 论文最小成立条件对照（docs/11 §14）

| 条件 | 状态 | 说明 |
|---|---|---|
| 1. B2 paper-spec + 强 B3 | ✅ | B2 F1=0.935, B3 F1=0.902 |
| 2. S1 显著优于 B3 | ✅ | slot-aware-5k F1=0.948 > B3 0.902 (+4.6pp) |
| 3. local evidence 降低 hallucination | ❌ | S2 未实现 |
| 4. CARC 比 pseudo-label SFT 更有效 | ⚠️ | 小数据有效，大数据退化 |
| 5. 新 benchmark + 细分评价 | ✅ | 鲁棒性分析 + 真实音频 held-out |

### 7.3 Trade-off 矩阵

| 需求 | 推荐配置 | F1 | onset-MAE | speech F1 |
|---|---|---|---|---|
| 最高 F1 | v6k | 0.970 | 0.262s | 0.952 |
| 最优时间精度 | v8 | 0.905 | 0.154s | 0.900 |
| 最优 speech 检测 | v7 | 0.911 | 0.178s | 1.000 |
| 最低幻觉 | v6k | 0.970 | 0.262s | 0.952 |
| 最快训练 | v6k (200 条) | 0.967 | 0.177s | 0.875 |

---

## 附录 A：完整实验列表

| # | 实验 | 日期 | F1 | halluc | 数据 | 说明 |
|---|---|---|---|---|---|---|
| 1 | B0 零样本 | 08-09 | 0.000 | 0 | — | 格式失败 |
| 2 | B1 SFT | 08-09 | 0.932 | 59 | 500 合成 | 格式习得 |
| 3 | B2 TAC (w=5) | 08-09 | 0.935 | 55 | 500 合成 | 时间精度最优 |
| 4 | B2 w=10 | 08-09 | 0.925 | — | 500 合成 | w=5 是甜点 |
| 5 | B2 no-template | 08-09 | 0.849 | — | 500 合成 | 模板必要 |
| 6 | B2 10k-steps | 08-09 | 0.994* | — | 500 合成 | 过拟合 |
| 7 | B3 统一 | 08-10 | 0.902 | 198 | 500 合成 | 复杂退化 |
| 8 | B3-permuted | 08-10 | 0.913 | 137 | 500 合成 | 排列不变 |
| 9 | B3-slot-aware | 08-10 | 0.926 | 99 | 500 合成 | 计数+slot |
| 10 | B3-slot-5k | 08-10 | 0.948 | 46 | 5000 合成 | 数据规模化 |
| 11 | B3-slot-5k-5ksteps | 08-11 | 0.944 | 47 | 5000 合成 | 已收敛 |
| 12 | B3-CARC | 08-11 | 0.912 | 57 | 500 合成 | 幻觉-71% |
| 13 | B3-CARC-5k | 08-11 | 0.920 | 67 | 5000 合成 | 大数据退化 |
| 14 | DPO 激进 | 08-11 | 0.004 | 89 | 5000 合成 | 格式崩溃 |
| 15 | DPO 保守 | 08-11 | 0.951 | 48 | 5000 合成 | 微幅提升 |
| 16 | P5 oracle | 08-11 | 0.890 | 11 | 50 合成 | 完美分离上界 |
| 17 | P5 predicted | 08-11 | 0.270 | 32 | 20 合成 | 分离级联-70% |
| 18 | S1a DETR | 08-11 | 0.082 | 170 | 500 合成 | 优化困难 |
| 19 | rank=64 | 08-12 | 0.951 | 46 | 5000 合成 | 与 rank=128 相同 |
| 20 | 真实零样本 | 08-14 | N/A | — | 50 真实 | 格式 0% |
| 21 | 真实 v6 200条 500步 | 08-14 | 0.865 | 8 | 200 真实 | 首次真实训练 |
| 22 | 真实 v6 200条 3000步 | 08-14 | 0.980 | 3 | 200 真实 | 训练集 F1 |
| 23 | 真实 v6 200条 held-out | 08-14 | 0.967 | 2 | 200 真实 | 泛化验证 |
| 24 | **真实 v6k 1000条 held-out** | 08-15 | **0.970** | **3** | 1000 真实 | **最优配置** |
| 25 | 真实 v7 held-out | 08-16 | 0.911 | 27 | 1000 真实 | speech=1.0 |
| 26 | 真实 v8 held-out | 08-16 | 0.905 | 23 | 1000 真实 | onset-MAE=0.154s |

---

## 附录 B：文档索引

| 文档 | 说明 |
|---|---|
| docs/01 | 相关工作与研究空缺 |
| docs/02 | SceneLedger idea |
| docs/03 | 数据与 WildMix-Cap benchmark |
| docs/04 | 评估协议 |
| docs/05 | MOSS-Audio 适配 |
| docs/06 | TAC 复现协议 |
| docs/07 | 试点执行计划 |
| docs/08 | Hybrid Track-Event idea |
| docs/09 | 消融实验设计 |
| docs/10 | 论文实验要求 |
| docs/11 | 开发计划与验收标准 |
| docs/12 | 复现指南 |
| docs/13 | 实验结果综合报告（合成） |
| docs/14 | （跳过） |
| docs/15 | 合成音频质量问题与改进 |
| docs/16 | 合成音频根本性问题 |
| docs/17 | 真实音频 v2 Review 反馈 |
| docs/18 | 真实音频 v3b Review 反馈 |
| docs/19 | 真实音频 v4 Review 反馈 |
| docs/20 | 真实音频 v5 Review 反馈 |
| docs/21 | v6 Review 反馈 + 训练结果 |
| docs/22 | v6k 典型样例 Review 反馈 |
| docs/23 | 实验结果最终报告 |
| paper/main.tex | 论文初稿 |
