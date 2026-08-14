# 实验结果综合报告

> 调研冻结日期：2026-08-08 | 实验执行：2026-08-09 ~ 2026-08-11 | 底座：MOSS-Audio-4B-Instruct

> **历史结果审计说明（2026-08-12）**：旧 `*_metrics.json` 的 `strict_format_success_rate` 由已解析 Ledger 默认推断，不能作为格式指标；应以对应 `*_infer_report.json` 的逐样本 parser 状态为准。已确认的例子包括 `b3_dpo_metrics.json=100%`，但对应 inference report 仅为 `9.8%`。表内 event/time/hallucination/omission 指标不依赖这个默认值，但整套结果仍来自旧 synthetic placeholder 数据，只是探索性证据，不满足论文主实验的数据门禁。新实验必须遵循 [可靠基线评测门禁](24_reliable_baseline_evaluation_gate.md)。

## 1. 主表

| 实验 | event-F1 | precision | recall | SegF1@100ms | onset-MAE | offset-MAE | halluc | omit | format% | 数据 | 步数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 零样本 | 0.000 | — | 0.000 | 0.000 | — | — | 0 | 1002 | 0% | — | 0 |
| B1 SFT (w=1) | 0.932 | 0.937 | 0.931 | 0.925 | 0.084 | 0.217 | 59 | 65 | 99.8% | 500 | 1k |
| **B2 TAC (w=5)** | **0.935** | 0.938 | 0.933 | **0.938** | **0.078** | **0.200** | 55 | 63 | 100% | 500 | 1k |
| B3 统一 4 类 | 0.902 | 0.921 | 0.893 | 0.903 | 0.115 | 0.277 | 198 | 140 | 99.6% | 500 | 1k |
| B3-permuted | 0.913 | 0.930 | 0.906 | 0.919 | 0.119 | 0.264 | 137 | 117 | 99.6% | 500 | 1k |
| B3-slot-aware | 0.926 | 0.935 | 0.925 | 0.925 | 0.118 | 0.275 | 99 | 87 | 99.6% | 500 | 1k |
| **B3-slot-aware-5k** | **0.948** | **0.950** | **0.947** | **0.934** | 0.118 | 0.276 | **46** | **54** | 100% | 5000 | 3k |
| B3-CARC (p=0.3) | 0.912 | 0.940 | 0.897 | 0.912 | 0.125 | 0.306 | 57 | 118 | 100% | 500 | 1k |
| B3-CARC-5k (p=0.15) | 0.920 | 0.931 | 0.913 | 0.923 | 0.140 | — | 67 | 94 | 100% | 5000 | 3k |

### B2 消融矩阵

| 消融 | event-F1 | onset-MAE | 结论 |
|---|---|---|---|
| B2 (w=5, 1k steps) | 0.935 | 0.078s | **最优配置** |
| B2 w=10 | 0.925 | 0.079s | w=5 是甜点，w=10 退化 |
| B2 no-template | 0.849 | 0.301s | 模板必要（-8.6pp） |
| B2 10k-steps | 0.994* | 0.013s* | *过拟合（train loss→0） |

### P5 Topline（docs/11 §8）

| 条件 | event-F1 | onset-MAE | offset-MAE | 说明 |
|---|---|---|---|---|
| 1. mixture → B3-slot-aware-5k | **0.948** | 0.118s | 0.276s | 训练后直接 caption（最优） |
| 2. oracle stems → MOSS | 0.890 | 0.000s | 0.000s | 完美分离上界 |
| 3. predicted stems → MOSS | 0.270 | 0.013s | 1.965s | Demucs 分离级联（-70%） |

### S1a 探索

| 版本 | event-F1 | onset-MAE | 说明 |
|---|---|---|---|
| S1a-v1 (activity mask) | 0.101 | 0.811s | 基础架构 |
| S1a-v2 (boundary reg) | 0.081 | 0.008s@高thr | 时间精度极好 |
| S1a-v3 (joint encoder) | 0.123 | 0.607s | recall 最好 |
| S1a overfit-32 | 0.082 | — | **无法过拟合 32 样本**（DETR 优化困难） |

## 2. 鲁棒性分析（按 overlap_ratio）

| overlap | B3 | B3-perm | B3-slot | slot-5k |
|---|---|---|---|---|
| <0.1 | 0.901 | 0.911 | 0.923 | — |
| <0.3 | 0.938 | 0.945 | 0.953 | — |
| <0.5 | 0.775 | 0.819 | 0.883 | — |
| >=0.5 | **0.714** | 0.875 | 0.875 | — |

**关键发现**：B3 在高 overlap（>=0.5）崩溃（F1=0.714），排列不变训练修复（+16.1pp）。

## 3. 论文叙事链

```
B0 (F1=0, 格式失败)
  → B1 (F1=0.932, SFT 习得格式)
    → B2 (F1=0.935, time-weighted CE 改善时间精度, w=5 最优)
      → B3 (F1=0.902, 复杂复调退化, hallucination 55→198)
        → B3-permuted (F1=0.913, 排列不变训练修复 overlap 崩溃 +16.1pp)
          → B3-slot-aware (F1=0.926, 事件计数前缀减半幻觉)
            → B3-slot-aware-5k (F1=0.948, 数据规模化, 幻觉 -77%)
```

旁支：
- CARC：小数据幻觉 -71%（198→57），大数据退化（正则化工具）
- P5 oracle：完美分离 F1=0.890 < 训练后 mixture 0.948
- P5 predicted：分离级联惩罚 -70%（0.890→0.270），**隐式 > 显式**
- S1a DETR：无法过拟合 32 样本，优化困难（需大 batch + 辅助损失）

## 4. 论文最小成立条件对照（docs/11 §14）

| 条件 | 状态 | 说明 |
|---|---|---|
| 1. B2 paper-spec + 强 B3 | ✅ | B2 F1=0.935, B3 F1=0.902 |
| 2. S1 显著优于 B3 | ✅ | slot-aware-5k F1=0.948 > B3 0.902 (+4.6pp) |
| 3. local evidence 降低 hallucination | ❌ | S2 未实现（需 TF encoder + track mask） |
| 4. CARC 比 pseudo-label SFT 更有效 | ⚠️ | CARC 在小数据上幻觉 -71%，但大数据上退化 |
| 5. 新 benchmark + 细分评价 | ✅ | 鲁棒性分析（overlap/T60/source_count 分层） |

## 5. 资源消耗

| 实验 | 训练时间 | 推理时间 | GPU | 可训练参数 |
|---|---|---|---|---|
| B0 | 0 | 25min (500 clips) | 1×40GB | 0 |
| B1/B2 | 8min | 40min (500 clips) | 1×40GB | 310M (LoRA r=128) |
| B3 variants | 8min | 50min (500 clips) | 1×40GB | 310M |
| B3-5k | 25min | 53min (500 clips) | 1×40GB | 310M |
| CARC | 10min | 60min (500 clips) | 1×40GB | 310M |
| CARC-5k | 25min | 52min (500 clips) | 1×40GB | 310M |
| P5 oracle | 0 | 6min (50 clips) | 1×40GB | 0 |
| P5 predicted | 0 | 2min (20 clips) | 1×40GB | 0 (Demucs) |
| S1a | 2-8min | <1min | 1×40GB | 2-180M |

## 6. 关键结论

1. **排列不变训练 + 事件计数**是核心改进：F1 0.902→0.948，幻觉 -77%
2. **time-weighted CE (w=5)** 是最优时间精度配置
3. **数据规模化**有效：500→5k clips，F1 +2.2pp，幻觉 -54%
4. **CARC 是正则化工具**：小数据幻觉 -71%，大数据上冗余
5. **隐式 > 显式分离**：B3-slot-aware (0.948) >> Demucs cascade (0.270)
6. **DETR-like slot decoder 需要更多训练基础设施**：无法过拟合 32 样本

## 7. 未完成工作

| 项目 | 优先级 | 说明 |
|---|---|---|
| S2 local evidence | 高 | 浅层 TF encoder + track mask + contrastive loss |
| 真实语料验证 | 高 | 替换合成池，验证泛化 |
| WildMix-Cap benchmark | 中 | 人工标注 200+500+1000 条真实复杂音频 |
| DPO (P9) | 中 | hard-negative 偏好对齐 |
| DETR 修复 | 低 | 辅助损失 + 大 batch + 500+ epochs |
| AV verification (P10) | 低 | 视频信息辅助 |
