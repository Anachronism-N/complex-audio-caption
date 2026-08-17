# v8 结果复核与 LLM 混音规划配对盲听

更新日期：2026-08-17

## 1. 结论先行

远端新增的 v8 证明了一个诊断现象：使用 RMS activity 修正时间标签后，模型的
onset MAE 从 v6k 的 0.262 秒下降到 0.154 秒，但 event F1 降到 0.905，且
hallucination 增加到 23。

这些数字不能作为论文级 held-out 结论。v8 可以保留为 exploratory diagnostic，
但不能用来判断 LLM mixer、数据扩展或模型架构是否有效。下一项实验应回到通过
catalog、stem、split 和人工听审约束的数据链路，先验证 LLM 规划是否真的比规则
规划生成了更合理的声音组合与时间关系。

## 2. 为什么 v8 不是论文级 held-out 实验

对 `configs/model/b3_real_v8_3k.yaml` 执行当前 CPU preflight：

```bash
sceneledger-train-preflight \
  --config configs/model/b3_real_v8_3k.yaml \
  --repo-root . \
  --output /tmp/v8_training_preflight.json
```

结果为 `authorized_to_train=false`、`publication_eligible=false`。具体问题是：

1. 2671 个 source 全部使用 `real:speech`、`real:music`、`real:sfx`、
   `real:ambience` 这四种占位 path，没有原始文件身份；
2. manifest 没有 mixture hash、stem hash、activity hash 和独立 stem；
3. 没有 train/val/test split contract、data gate 或 human-audit binding；
4. source caption 的 provenance 是 `model_prediction`，review CSV 中已经能看到
   `There is a silence.`、把 speech 描述成 orchestral music、把 restaurant ambience
   描述成高速汽车等错误；
5. speech 只有两个 MOSS demo 文件，GTZAN music 也可能含人声，无法支持严格的
   source-disjoint 泛化结论；
6. 所谓 held-out 指标使用的是 `rv8_0901` 到 `rv8_1000`。按训练代码的实际
   `group_split(seed=20260816)` 重放后，这 100 条中有 95 条属于训练 membership，
   只有 5 条属于内部 validation；
7. inference report 没有 manifest hash、dataset ID、expected split、split contract
   或 prediction hash，无法证明评测输入。

因此“v6k F1 最好、v8 onset 最好”只能描述当前 legacy pipeline 的观察结果，不能
写成论文中的可靠消融结论。更不能据此继续手工构造 v9。

## 3. 下一实验回答的唯一问题

> 在完全相同的冻结候选 source slate、模板、时长和渲染器下，LLM 选择具体音源并
> 规划 onset，是否比确定性规则选择产生更合理的场景共现与时间结构，同时不损害
> source audibility、caption support、timestamp alignment 和 naturalness？

这里不训练 caption 模型。先对数据生成策略做因果清晰的 matched A/B 验证；只有
该实验通过，才扩大数据规模并比较下游模型。

## 4. 新增盲听工具的约束

`sceneledger-mixture-review` 会在生成 review sheet 前强制检查：

- Rule 和 LLM recipes 数量、顺序、seed、template、duration、candidate task hash
  完全相同；
- `--rule-recipes` 确实来自 rule planner，`--llm-recipes` 确实来自 LLM compiler；
- 每个 rendered scene 都绑定正确的 recipe plan SHA256；
- 两个 `mixture_quality.json` 都必须为 PASS，并且其中的 manifest SHA256 必须与
  当前输入 manifest 完全一致；
- 实际 source ID 和 onset 与 `source_plan` 完全一致；
- 每个 source 都有 isolated stem、stem hash 和 activity hash；
- mixture/dry-mixture hash 非空，Ledger 可验证；
- review package 中 mixture 和 stem 被复制成匿名 A/B 文件，复制前后文件 SHA256
  一致；
- A/B assignment 使用 prepare 时生成的私有随机 salt，并强制保持两侧数量平衡；
- reviewer 不能看到 Rule/LLM assignment；private key 被单独保存并绑定哈希；
- 汇总时再次检查 package、CSV immutable fields、metadata 和 key 是否被修改。

只要输入还是 v8 这种无 stem 的 compatibility manifest，prepare 会直接失败。

## 5. 生成 120 对匿名 review package

前置步骤按照 `docs/39_llm_source_and_timeline_planner.md` 完成：使用同一 tasks
生成 `rule_recipes.jsonl` 和 `llm_recipes.jsonl`，通过 compare 后分别渲染。

```bash
export RUN_ROOT=/data/sceneledger_runs/llm_source_timeline_v1

sceneledger-mixture-review prepare \
  --rule-recipes "$RUN_ROOT/recipes/rule_recipes.jsonl" \
  --rule-manifest "$RUN_ROOT/rendered/rule/test/manifest.jsonl" \
  --rule-quality-report "$RUN_ROOT/rendered/rule/mixture_quality.json" \
  --rule-audio-base "$RUN_ROOT/rendered/rule/test" \
  --llm-recipes "$RUN_ROOT/recipes/llm_recipes.jsonl" \
  --llm-manifest "$RUN_ROOT/rendered/llm/test/manifest.jsonl" \
  --llm-quality-report "$RUN_ROOT/rendered/llm/mixture_quality.json" \
  --llm-audio-base "$RUN_ROOT/rendered/llm/test" \
  --package-dir "$RUN_ROOT/review/blind_package" \
  --sample-count 120 \
  --seed planner-review-20260817 \
  --output-csv "$RUN_ROOT/review/blind_review.csv" \
  --output-metadata "$RUN_ROOT/review/blind_review.metadata.json" \
  --output-key "$RUN_ROOT/private/blind_review.key.json"
```

发给 reviewer 的内容只有：

- `blind_review.csv` 的独立副本；
- `blind_package/`。

不要发送 `blind_review.key.json`，也不要把 Rule/LLM 原始目录名告诉 reviewer。
CSV 中的音频和 stem 路径相对于 `blind_package/`。

## 6. Reviewer 具体填写内容

每个 reviewer 独立听完 A/B mixture，并按需听匿名 isolated stems。每侧填写 1–5 分：

- `all_sources_audible`：expected events 中的 source 是否在 mixture 中可听；
- `scene_plausibility`：这些声音在同一真实场景中共现是否合理；
- `temporal_plausibility`：出现顺序、进入时机和重叠是否合理；
- `naturalness`：音量、混响、淡入淡出、ducking 和整体听感；
- `caption_support`：expected event 文本是否被对应 stem/mixture 支持；
- `timestamp_alignment`：标注 span 与可听 activity 是否一致。

还需填写：

- `inaudible_sources_count`；
- `unsupported_labels_count`；
- `preference_a_b_tie`，只能是 `a`、`b` 或 `tie`；
- reviewer、UTC 时间和必要 notes。

建议至少 2 名 reviewer，每人使用相同任务但不同 CSV 副本。不要共同商议答案。

## 7. 验证、解盲和 Go/No-Go

```bash
sceneledger-mixture-review summarize \
  --review-csv "$RUN_ROOT/review/reviewer_1.csv" \
  --review-csv "$RUN_ROOT/review/reviewer_2.csv" \
  --metadata "$RUN_ROOT/review/blind_review.metadata.json" \
  --key "$RUN_ROOT/private/blind_review.key.json" \
  --min-plausibility-delta 0.25 \
  --max-safety-regression 0.10 \
  --max-sign-p 0.05 \
  --output "$RUN_ROOT/review/llm_vs_rule_summary.json"
```

退出码为 0 表示 `go_for_scale=true`；退出码 3 表示 review 完整、制品有效，但结果
没有达到扩大实验条件。默认 Go 条件同时要求：

1. LLM 的 scene 和 temporal plausibility 平均各提高至少 0.25 分；
2. audibility、naturalness、caption support、timestamp alignment 的任一平均退化
   不超过 0.10 分；
3. LLM 的 inaudible source 和 unsupported label 总数不高于 Rule；
4. 样本级 consensus 中 LLM 胜过 Rule，双侧 sign test `p <= 0.05`。

如果 No-Go，不训练新 caption 模型。先查看是候选 caption 质量、LLM 选源、时间规划
还是 renderer 参数导致失败。只有 Go 才扩大到数千 scene，并在冻结 source-disjoint
test 上进行下游训练消融。

## 8. 该实验能证明和不能证明什么

通过后可以证明：在给定 audited candidate slate 的条件下，LLM planner 相比规则
planner 改善了混音数据的场景与时间合理性，且没有牺牲声学证据质量。

它仍不能证明最终 caption 模型优于 baseline，也不能证明 text-only LLM 真正理解了
waveform。当前 LLM 根据 catalog caption/metadata 选源；声音真值来自 stem。后续可在
同一合同下加入 audio-capable reranker，形成 Rule、text-LLM、audio-LLM 三臂实验。
