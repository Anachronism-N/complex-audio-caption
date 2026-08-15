# 下一步实验：冻结 Test 上的盲法模型语义评审

更新日期：2026-08-15

## 1. 为什么新增这一步

自动 event-F1 主要检查事件类型和时间重叠；caption token-F1 又会惩罚合理的
同义改写。它们都不能独立回答以下问题：

- 描述中的声音、说话人属性和声学环境是否真的可听见；
- 模型是否遗漏了被遮蔽但仍可辨认的事件；
- 模型是否写出了音频不支持的情感、场所、乐器或动作；
- tuned 模型是真的理解更好，还是只学会输出七事件格式和时间模板。

远端 `3a1731a` 新增的 20 条 review 包不能回答这些问题。其 20 条中有 15 条
被训练使用，所谓 GT 是 `model_prediction`，reviewer 同时看到 GT 和模型名称，
而生成脚本依赖仓库中不存在的 predictions 与 `/tmp/heldout_manifest.jsonl`。
旧 CSV 已更名为 `heldout_review_diagnostic_invalid.csv`，只能用于旧数据排错。

本轮新增 `sceneledger-model-review`，在通过认证的 Real-Complex test 上对
zero-shot 和 B3-tuned 做随机盲法 A/B 听审。这是模型实验的语义裁决层，不替代
训练前的 mixture/source 审核。

## 2. 评审包的强制前置条件

`prepare` 只有在以下条件全部成立时才生成任务：

1. test manifest 与 data gate、split contract 的 SHA-256 一致；
2. zero-shot 和 tuned predictions 都完整覆盖冻结 test，不能使用 `--limit`；
3. 两份 inference report 的 dataset ID、test split、sample IDs 和 prediction
   hash 均匹配；
4. 两份 inference report 都是 v2，且每条样本都有显式 track 完整性的 parser 状态；
5. tuned 的 `validity_audit.json` 为 `certified_generalization`，并绑定同一份
   tuned inference report；
6. 被抽中的 mixture 音频实际存在。

任务按 template round-robin 分层抽样，默认 60 条。对每条样本用冻结 seed
随机交换 A/B；CSV 不暴露 arm，映射只在 `.key.json` 中。候选保存完整结构化
event/type/span/text，而不是截断到 40 字符。

## 3. 自动生成

完整运行三折 anchor：

```bash
bash scripts/run_b3_real_complex_anchor.sh "$EXP_ROOT" "$MOSS_WEIGHTS" 1000
```

在 zero-shot、训练、tuned、结果认证和 comparison 均完成后，脚本自动生成：

```text
$EXP_ROOT/evaluation/human_model_review.csv
$EXP_ROOT/evaluation/human_model_review.metadata.json
$EXP_ROOT/evaluation/human_model_review.key.json
```

若只需从已有的有效结果重建任务，可运行：

```bash
sceneledger-model-review prepare \
  --manifest "$EXP_ROOT/test/manifest.jsonl" \
  --audio-base "$EXP_ROOT/test" \
  --zero-predictions "$EXP_ROOT/evaluation/zero_shot/predictions.jsonl" \
  --zero-inference-report "$EXP_ROOT/evaluation/zero_shot/inference_report.json" \
  --tuned-predictions "$EXP_ROOT/evaluation/b3_tuned/predictions.jsonl" \
  --tuned-inference-report "$EXP_ROOT/evaluation/b3_tuned/inference_report.json" \
  --validity-audit "$EXP_ROOT/evaluation/b3_tuned/validity_audit.json" \
  --split-contract "$EXP_ROOT/gate/split_contract.json" \
  --data-gate-summary "$EXP_ROOT/gate/experiment_data_summary.json" \
  --sample-count 60 \
  --output-csv "$EXP_ROOT/evaluation/human_model_review.csv" \
  --output-metadata "$EXP_ROOT/evaluation/human_model_review.metadata.json" \
  --output-key "$EXP_ROOT/evaluation/human_model_review.key.json"
```

## 4. Reviewer 怎么填写

在分发前复制两份 CSV，分别交给两个独立 reviewer：

```bash
cp "$EXP_ROOT/evaluation/human_model_review.csv" \
   "$EXP_ROOT/evaluation/human_model_review.reviewer1.csv"
cp "$EXP_ROOT/evaluation/human_model_review.csv" \
   "$EXP_ROOT/evaluation/human_model_review.reviewer2.csv"
```

reviewer 不得打开 key，不显示 reference，不讨论前一位 reviewer 的答案。每条先
完整听音频，再分别评 A/B：

| 字段 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| semantic support | 大部分内容不可听证 | 核心声音对但属性有误 | 所有主要断言均有声学支持 |
| completeness | 多个主要事件遗漏 | 只漏弱/短事件 | 所有可辨事件均覆盖 |
| temporal alignment | 边界/顺序明显错误 | 大致正确但有可闻偏移 | onset/offset 与听感一致 |
| source attribution | speaker/track 多处混淆 | 部分归属不确定 | 归属全部正确 |

无 attribution 问题时该字段填 `na`。hallucination 是“caption 断言存在但音频
不支持”的事件/属性数；omission 是“音频中明确可辨但 caption 未覆盖”的事件数。
最后填写 `a`、`b` 或 `tie`，不要根据文本长度或格式整齐程度偏好某一候选。

不得修改 `review_id` 到 `candidate_b_json` 的不可变列，否则汇总会拒绝。时间和
reviewer 必填；建议使用 UTC ISO-8601，例如 `2026-08-15T08:00:00Z`。

## 5. 校验、揭盲和汇总

两个 reviewer 完成后运行：

```bash
sceneledger-model-review summarize \
  --review-csv "$EXP_ROOT/evaluation/human_model_review.reviewer1.csv" \
  --review-csv "$EXP_ROOT/evaluation/human_model_review.reviewer2.csv" \
  --metadata "$EXP_ROOT/evaluation/human_model_review.metadata.json" \
  --key "$EXP_ROOT/evaluation/human_model_review.key.json" \
  --output "$EXP_ROOT/evaluation/human_model_review.summary.json"
```

汇总前会验证 CSV schema、任务顺序、候选文本 hash、key hash 和 review ID，然后
才揭盲。输出包括：

- 两个 arm 的 semantic/completeness/time/attribution 平均分；
- hallucination/omission 总数；
- 每个评分的 tuned-minus-zero 配对差值；
- tuned/zero/tie 的逐 reviewer judgment 数、逐样本共识数，以及基于逐样本共识的
  two-sided sign-test p 值；
- reviewer 偏好的逐样本 pairwise agreement；
- `go_for_scale` 与三个可解释 decision checks。

第一轮 GO 标准为：语义、完整性和时间平均差值均不下降，tuned 幻觉和遗漏均
不增加，且 tuned 偏好次数大于 zero-shot。sign-test 在 60 条 pilot 上主要作为效应稳定性
诊断，不强行要求显著；论文正式表建议扩大人工 test 或增加 reviewer。

## 6. 当前执行顺序

这一步不能先于数据流程。当前服务器顺序仍然是：

```text
三套 source audit
  -> recipe review
  -> 三折 render/data gate
  -> 60 条 mixture 人工审核
  -> zero-shot / train / tuned
  -> result certification
  -> 60 条盲法模型语义评审
```

在前三个 source catalog 和人工 mixture gate 尚未通过前，不需要运行本页命令，
也不要继续 review 旧 v6 的 20 条样本。
