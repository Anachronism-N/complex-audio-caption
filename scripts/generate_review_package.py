"""Generate a human review package: 50 clips with GT, predictions, and error analysis.

Outputs:
- data/review/review_50.csv — review-ready CSV (open in Excel/Sheets)
- data/review/README.md — review instructions
"""
import json, csv, random
from pathlib import Path

# Load manifest + predictions
entries = []
with open("data/derived/b3_5k/manifest.jsonl") as f:
    for line in f:
        entries.append(json.loads(line))

preds = {}
with open("reports/b3_slot_aware_5k_predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        preds[d["sample_id"]] = d

# Select 50 diverse clips (5 per template, prioritizing errors)
template_clips = {}
for e in entries:
    t = e["scene"]["template"]
    sid = e["scene"]["scene_id"]
    if sid not in preds:
        continue
    has_error = False
    gt_events = set((ev["type"], round(ev["spans"][0]["start_sec"], 1),
                     round(ev["spans"][-1]["end_sec"], 1)) for ev in e["target_ledger"]["events"])
    pred_events = set((ev["type"], round(ev["spans"][0]["start_sec"], 1),
                       round(ev["spans"][-1]["end_sec"], 1)) for ev in preds[sid]["events"])
    if gt_events != pred_events:
        has_error = True
    template_clips.setdefault(t, []).append((sid, e, preds[sid], has_error))

selected = []
rng = random.Random(42)
for t in sorted(template_clips):
    clips = template_clips[t]
    # prioritize error clips
    error_clips = [c for c in clips if c[3]]
    correct_clips = [c for c in clips if not c[3]]
    rng.shuffle(error_clips)
    rng.shuffle(correct_clips)
    # take up to 2 error + 3 correct per template
    take = error_clips[:2] + correct_clips[:3]
    selected.extend(take[:5])

selected = selected[:50]
print(f"Selected {len(selected)} clips for review")

# Generate review CSV
out_dir = Path("data/review")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "review_50.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow([
        "clip_id", "template", "audio_path", "duration",
        "gt_n_events", "gt_events", "pred_n_events", "pred_events",
        "error_type", "n_halluc", "n_omit", "n_correct",
        "human_label_correct", "human_notes"
    ])
    for sid, entry, pred, has_error in selected:
        gt_evs = entry["target_ledger"]["events"]
        pred_evs = pred["events"]
        gt_set = set((e["type"], round(e["spans"][0]["start_sec"], 1),
                      round(e["spans"][-1]["end_sec"], 1)) for e in gt_evs)
        pred_set = set((e["type"], round(e["spans"][0]["start_sec"], 1),
                        round(e["spans"][-1]["end_sec"], 1)) for e in pred_evs)
        n_correct = len(gt_set & pred_set)
        n_halluc = len(pred_set - gt_set)
        n_omit = len(gt_set - pred_set)
        error_type = "correct" if not has_error else (
            "hallucination" if n_halluc > 0 and n_omit == 0 else
            "omission" if n_omit > 0 and n_halluc == 0 else
            "mixed"
        )
        gt_str = " | ".join(f'{e["type"]} [{e["spans"][0]["start_sec"]:.1f}-{e["spans"][-1]["end_sec"]:.1f}] {e["text"][:30]}' for e in gt_evs)
        pred_str = " | ".join(f'{e["type"]} [{e["spans"][0]["start_sec"]:.1f}-{e["spans"][-1]["end_sec"]:.1f}] {e["text"][:30]}' for e in pred_evs)
        w.writerow([
            sid, entry["scene"]["template"],
            f"audio/{sid}.wav", entry["scene"]["duration"],
            len(gt_evs), gt_str,
            len(pred_evs), pred_str,
            error_type, n_halluc, n_omit, n_correct,
            "", ""  # human fills in
        ])

print(f"Wrote {csv_path}")

# Generate README
readme = out_dir / "README.md"
readme.write_text("""# 人工 Review 指南

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
""", encoding="utf-8")

print(f"Wrote {readme}")
print(f"\nReview package ready at data/review/")
print(f"Open {csv_path} in Excel/Sheets to start reviewing.")
