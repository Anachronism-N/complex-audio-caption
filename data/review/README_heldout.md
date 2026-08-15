# real_mix_v6 Review 包：仅供历史诊断

## 重要更正

这个 review 包**不是 held-out 泛化评测**，不能用于验证模型在“未训练真实音频”
上的表现：

- `rv6_0181`--`rv6_0200` 中有 15 条实际被训练器使用；
- 15 条训练样本的 event-F1=1.000、onset MAE=0；
- 仅 5 条 sample-ID 未见，其 event-F1=0.867、onset MAE=0.710 s；
- manifest 将原始声源替换为 `real:speech`、`real:sfx` 等占位符，所以连这
  5 条也不能证明 source-disjoint；
- 所谓 GT 的 provenance 是 `model_prediction`，不是人工标注。

机器审计见 `reports/b3_real_v6_3k_heldout_validity_audit.json`，完整说明见
`docs/34_v6_heldout_forensics_and_result_certification.md`。

原 CSV 已重命名为：

```text
data/derived/real_mix_v6/heldout_review_diagnostic_invalid.csv
```

它只可用于检查旧 v6 的音频/pseudo-label 质量，不能汇入论文指标或模型对比。
旧 `scripts/gen_heldout_review.py` 现在会直接退出，防止再次生成错误结论。

## 正确的模型人工评审

Real-Complex 三折实验通过数据、人工听审和结果认证后，
`scripts/run_b3_real_complex_anchor.sh` 会自动生成：

```text
$EXP_ROOT/evaluation/human_model_review.csv
$EXP_ROOT/evaluation/human_model_review.metadata.json
$EXP_ROOT/evaluation/human_model_review.key.json
```

reviewer 只能打开 CSV 和音频，不应打开 key。候选 A/B 已随机交换，分别对
语义支持度、完整性、时间对齐、source attribution、幻觉和遗漏评分。
填写完毕后运行：

```bash
sceneledger-model-review summarize \
  --review-csv "$EXP_ROOT/evaluation/human_model_review.csv" \
  --metadata "$EXP_ROOT/evaluation/human_model_review.metadata.json" \
  --key "$EXP_ROOT/evaluation/human_model_review.key.json" \
  --output "$EXP_ROOT/evaluation/human_model_review.summary.json"
```

summary 会在校验任务哈希后才揭盲，分别报告 zero-shot/B3 的平均评分、幻觉与
遗漏总数、配对差值、偏好胜负和 sign-test p 值。
