# Held-out Review 指南

## 目标
对比模型预测与 GT（人工标注），验证模型在**未训练过的真实音频**上的表现。

## 数据位置

| 文件 | 说明 |
|---|---|
| `data/derived/real_mix_v6/heldout_review.csv` | 20 条 held-out clip 的 GT vs 预测对比表 |
| `/tmp/real_mix_v6/audio/rv6_0181.wav` ~ `rv6_0200.wav` | 音频文件 |

## Review 结果预览

**20/20 事件数完全匹配**（gt_n_events == pred_n_events）！

## CSV 列说明

| 列 | 说明 |
|---|---|
| clip_id | 音频文件名 |
| audio_path | 音频路径 |
| duration | 时长 |
| scene_name | 场景模板 |
| gt_n_events | GT 事件数 |
| gt_events | GT 事件列表（类型 [时间] 文本） |
| pred_n_events | 模型预测事件数 |
| pred_events | 模型预测事件列表 |
| pred_format_ok | 格式是否成功 |
| match | correct/halluc/omit/mixed |
| errors | 正确/幻觉/遗漏计数 |
| notes | **人工填写** |

## Review 步骤

1. 打开 `heldout_review.csv`（Excel/Sheets）
2. 逐条听 `/tmp/real_mix_v6/audio/rv6_0181.wav` ~ `rv6_0200.wav`
3. 对比：
   - **GT 事件**：你听到的声音是否与 GT 描述匹配？
   - **预测事件**：模型预测是否与实际听到的声音匹配？
   - **时间精度**：预测的 onset/offset 是否大致准确？
4. 填写 `notes`：记录任何错误模式

## 重点关注

1. **事件数 100% 匹配**——模型能正确计数
2. **时间精度**——预测时间是否在 0.1-0.2s 范围内？
3. **文本质量**——预测描述是否准确？
4. **幻觉/遗漏**——虽然事件数匹配，但具体事件是否对齐？

## 统计摘要

- Held-out F1: 0.967
- 格式成功率: 100%
- 幻觉: 2
- 遗漏: 2
- Per-type F1: music=1.0, sfx=0.948, speech=0.875
