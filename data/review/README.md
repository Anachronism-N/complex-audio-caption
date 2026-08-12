# 人工 Review 指南

## 目标
检查 50 条合成音频的标签是否正确，以及模型预测是否合理。

## 步骤

### 1. 解压音频
```bash
# 如果音频还没解压
cd data/derived/b3_5k && tar xf audio.tar
# 音频在 data/derived/b3_5k/audio/mix_XXXXXX.wav
```

### 2. 打开 review CSV
```bash
# 用 Excel 或 Google Sheets 打开
data/review/review_50.csv
```

### 3. 逐条 review
对每条 clip：

1. **听音频**：打开 `audio/{clip_id}.wav`，听完整段
2. **检查 GT（ground truth）**：
   - `gt_events` 列列出标准答案（类型 [时间] 文本）
   - 确认：你听到的事件与 GT 是否匹配？
   - 标签错位示例：GT 说"speech"但你听到的是音乐
3. **检查预测**：
   - `pred_events` 列列出模型预测
   - 确认：模型预测是否合理？
4. **填写**：
   - `human_label_correct`：GT 是否正确？（Y/N/partial）
   - `human_notes`：任何发现（错位、遗漏、幻觉、时间不准等）

### 4. 关注重点
- **标签错位**：GT 标注的事件类型/时间是否与实际听到的不符？
- **幻觉**：模型预测了实际听不到的事件
- **遗漏**：模型漏掉了实际能听到的事件
- **时间精度**：预测的 onset/offset 是否准确？
- **文本质量**：描述文本是否准确反映音频内容？

### 5. Review 完成后
- 保存标注后的 CSV
- 统计：GT 正确率、模型错误类型分布
- 如果 GT 错误率 > 10%，renderer 有 bug 需修复
- 如果模型幻觉/遗漏有规律，记录模式用于改进

## 列说明
| 列 | 说明 |
|---|---|
| clip_id | 音频文件名（不含.wav） |
| template | 场景模板 |
| audio_path | 音频相对路径 |
| duration | 时长（秒） |
| gt_n_events | GT 事件数 |
| gt_events | GT 事件列表（类型 [时间] 文本） |
| pred_n_events | 模型预测事件数 |
| pred_events | 模型预测事件列表 |
| error_type | correct/hallucination/omission/mixed |
| n_halluc | 幻觉事件数 |
| n_omit | 遗漏事件数 |
| n_correct | 正确匹配事件数 |
| human_label_correct | 人工填写：GT 是否正确 |
| human_notes | 人工填写：备注 |
