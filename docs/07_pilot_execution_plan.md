# 首轮 Pilot 执行计划与验收表

## 1. Pilot 要回答什么

首轮不追求“训练出最终大模型”，只回答四个决定项目生死的问题：

1. MOSS-Audio 能否稳定学会 TAC-style 的 typed timestamp grammar？
2. TAC-style 数据和 weighted timestamp loss 能否在真实复杂音频上优于普通 SFT？
3. event slots 是否在 overlap/source-count 增加时比自回归序列更稳？
4. source-removal CARC 是否在真实数据降低 hallucination，而不是只学习 separator artifact？

如果前一个问题不成立，不并行扩大下一个阶段。

## 2. 八周建议节奏

### Week 1：环境、数据契约与 metric

- 锁定 MOSS-Audio upstream commit、模型 revision、CUDA/PyTorch/Transformers 版本；
- 下载 4B-Instruct，跑官方 inference 与一条官方 LoRA smoke run；
- 实现 ledger JSON ↔ XML-like caption round-trip；
- 实现 301 个 timestamp tokens 和 tokenizer save/load test；
- 实现 20 个 toy scenes 的 SegF1/EvtF1/boundary tests；
- 建立实验记录字段：git SHA、config hash、data manifest hash、seed、GPU、wall time。

交付：B0 输出、全绿单测、固定 metric container。没有这些，不开始大规模数据准备。

### Week 2：Dynamic Mixer R0/R1

- 实现 deterministic mixing、stem-level activity 与 manifest；
- 完成 6 个 paper-spec templates；
- 生成 500 条 R0 和首批 20k R1；
- 人工听检每个 template 至少 30 条；
- 检查 gain clipping、RIR tail、echo duplicate、speech truncation 与标签一致性；
- 冻结 validation/test mixtures。

交付：给定 seed 可逐样本复建，waveform/activity/hash 一致。

### Week 3：B1 静态 SFT

- MOSS-4B LoRA；
- 固定 brief/0.1 s；
- 训练到 parser success ≥99.5%；
- 输出 B0/B1 在 R0、R1-val、TACOS-val 的对比；
- 检查新增 token embedding 和 lm_head 确实更新。

交付：B1 checkpoint、训练曲线、parser error taxonomy。

### Week 4：B2 TAC-style

- dynamic style/merge/activity/resolution；
- weighted token CE；
- templates 与 TACOS train；
- 主 run 5k steps，至少先做一个 seed；
- 跑 `time_weight=1/5/10` 和 `no-template` 小消融。

交付：B1/B2 指标表。如果 B2 没有至少两项主要指标改善，停在本周调试。

### Week 5：真实 200 条 Pilot 与 B3

- 完成 200 条人工标注，至少双人标注 + 争议复核；
- 覆盖多说话、speech+music/sfx、music+lyrics、一般 polyphony、harsh acoustics；
- 标出 earliest/best/latest boundary、audibility 与不确定/不可辨认文本；
- B3 加入 overlapping speech 和 lyrics；
- 报告每个场景分层而非只给总分。

交付：冻结的 `pilot-v1` dev/test 切分、B0–B3 error dashboard。

### Week 6：S1 event ledger

- 从 MOSS `get_audio_features()` 接 last/deepstack；
- 实现 10 Hz temporal fusion、24 slots、Hungarian matching；
- 先只训练 type/null/activity/boundary，不急于完整文本；
- 在合成与 real pilot 上画 source count/overlap robustness curve；
- 可视化每个 slot activity，人工查 slot collapse/duplicate。

交付：S1 与 B3 的结构对比。如果只在合成集有增益，不继续堆 decoder。

### Week 7：S2 evidence-conditioned text

- matched slot masked pooling；
- 4 local prefix tokens + 2 gated global tokens；
- speech/lyrics/music/sfx adapters；
- null/confidence/source loss；
- 做 `no-local` 和 `unrestricted-global` 两个关键消融。

交付：unsupported-event rate、cpWER/lyrics CER 和 local-evidence 可视化。

### Week 8：CARC 最小闭环

- 先用真 stems 做 add/remove/shift，确认集合代数 loss 有效；
- 再对 5k–10k 真实视频运行 WhisperX/Demucs/MOSS/FLAM；
- SAM Audio 只处理 teacher 冲突或高价值子集；
- 对 residual leakage 做人工抽检；
- 比较 S2、S2+pseudo-SFT、S3-CARC。

交付：是否扩大到 50k–100k web clips 的 go/no-go 决策。

## 3. 200 条人工 Pilot 配额

同一 clip 可跨多个属性，但主层级配额互斥：

| 主层级 | 数量 | 关键标注 |
|---|---:|---|
| overlapping multi-speaker | 50 | speaker lane、重叠区 transcript、echo vs speaker |
| speech + music/sfx | 50 | 遮蔽、前后景、speech 可听度 |
| music + vocals/lyrics | 40 | lyric presence、line text、vocal/伴奏 overlap |
| general polyphony | 40 | repeated instances、short transients、source relation |
| harsh acoustics | 20 | noise/reverb/echo/codec；尽量与其他层级交叉 |

建议 150 dev + 50 internal test。200 条只能用于早期可证伪，不能作为最终 benchmark；一旦用于频繁调参，就不能继续声称是测试集。

## 4. 每个实验必须记录

```text
run_id
parent_run_id
git_commit
upstream_model_revision
config_sha256
train_manifest_sha256
eval_manifest_sha256
seed
GPU type/count
precision
effective batch
train steps / seen audio hours
best-checkpoint rule fixed before training
all primary and stratified metrics
parser failures and excluded samples
```

禁止只保存“best score”而丢失 selection rule。checkpoint 选择使用预先指定的 dev `ST-F1 - beta * Hal`，不能看 test 后重新加权。

## 5. Pilot 决策阈值

阈值是项目管理线，不是统计显著性结论：

- `B2 → proceed`：相对 B1，TACOS 与 real pilot 的 EvtF1/SegF1/Hal 三项各至少两项方向改善，且 parser valid ≥99.5%；
- `S1 → proceed`：real pilot ST-F1 至少有稳定正增益，且 source-count 曲线在 3+ sources 区域优于 B3；
- `S2 → proceed`：unsupported-event rate 相对 S1/B3 降低，同时 cpWER 或 lyrics CER 没有明显灾难性退化；
- `S3 → scale web data`：至少两个 seeds 下，S3 优于等量 pseudo-label SFT，并在 source-removal negative 与 harsh-acoustic slice 同时改善；
- 任一核心提升的 95% bootstrap CI 大范围跨 0：补数据/seed，不做结论。

## 6. 最小论文表格

主表至少包括：

| Model | ST-F1 | SegF1 | EvtF1@0.25/1.0 | Boundary MAE | Hal | cpWER | Lys CER | Robust AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSS zero-shot | | | | | | | | |
| B1 static SFT | | | | | | | | |
| B2 MOSS-TAC | | | | | | | | |
| B3 joint AR | | | | | | | | |
| S1 ledger | | | | | | | | |
| S2 + evidence | | | | | | | | |
| S3 + CARC | | | | | | | | |

必须另有：

- type/场景 slice 表；
- source count、SNR、T60、echo delay 曲线；
- CARC add/remove/shift 分解消融；
- qualitative ledger + activity heatmap；
- failure cases：不可听 lyrics、echo duplicate、separator leakage、semantic sibling confusion。

## 7. 代码实现优先级

严格按以下依赖顺序建立代码：

```text
schema/parser/metrics
  → deterministic mixer
  → MOSS data converter
  → B1 ordinary CE
  → B2 weighted CE + dynamic tasks
  → 200-real pilot
  → slot decoder
  → evidence text
  → CARC
```

不要先写 SAM Audio 大规模分离脚本或复杂 AV pipeline。没有 B2 和 200-real pilot，无法判断 pseudo-data 的提升是真实能力还是 teacher 自洽。

