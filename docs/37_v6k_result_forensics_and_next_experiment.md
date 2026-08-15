# v6k “F1=0.970”结果复核、指标修复与下一实验

更新日期：2026-08-15

## 1. 先给结论

远端提交 `6a00bb4` 新增的 B3-v6k 结果**只能用于诊断，不能作为 held-out
泛化结果或论文主表结果**。这不是因为 0.970 数值不够高，而是因为实验不满足
解释这个数字所需的前提：

1. 报告的 100 条 `rv6k_0901`--`rv6k_1000` 中，64 条被训练器实际用于训练；
2. 剩余 36 条只满足 sample ID 未见，原始 source identity 全部缺失，不能证明
   source-disjoint；
3. 评测是从 1,000 条训练 manifest 事后截取 100 条，不是运行前冻结并绑定合同的
   test manifest；
4. 原指标的 event-F1 只主要检查事件类型和时间重合，不检查 caption 文本是否正确；
5. 原 metrics 没有 `caption_token_f1`，且把 strict-format rate 写为 1.0，而原始
   inference report 实际是 0.98；
6. source-count MAE 为 1.14、event-to-source pointer accuracy 只有 0.34，说明高
   event-F1 没有同步转化为可靠的结构化声源解析。

本轮新增了两层 fail-closed 机制：一是论文结果必须使用
`sceneledger-metrics-v2`；二是可对已提交的 raw generation 做 CPU-only forensic
replay，恢复缺失的语义指标、解析证据和真实训练成员分组，但不会把污染实验追认为
论文结果。

## 2. 100 条结果究竟测到了什么

训练配置声明 `val_fraction=0.1`，但 v6k 只有若干占位 source 路径组合。训练器按这些
组合做 group split，实际成员为：

```text
训练 manifest:       1000
训练器实际 training:  702
内部 validation:      298
事后报告样本:          100
其中实际 training:      64
仅 sample-ID 未见:       36
```

使用提交中的 `raw_text` 和当前统一 parser/evaluator 离线复算后：

| 子集 | N | event-F1 | caption token-F1 | 100 ms Seg-F1 | onset MAE | format | source-count MAE | pointer acc. | omission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部事后子集 | 100 | 0.970 | 0.164 | 0.943 | 0.262 s | 0.980 | 1.140 | 0.340 | 9 |
| 实际训练见过 | 64 | 0.990 | 0.179 | 0.970 | 0.227 s | 1.000 | 1.141 | 0.362 | 2 |
| sample-ID 未见 | 36 | 0.935 | 0.137 | 0.895 | 0.325 s | 0.944 | 1.139 | 0.301 | 7 |

这里的 caption token-F1 是轻量、可确定性复算的词项重叠诊断，并不是完整语义指标；
同义改写会使它偏低。因此不能把 0.164 直接解释为“语义正确率 16.4%”。但它足以
证明原先的 0.970 没有评价 caption 内容：确有 event-F1 接近或等于 1.0、caption
token-F1 却只有约 0.07--0.08 的样本。正式实验还必须加入冻结 test 上的盲法人工语义
评审，不能仅依赖 token-F1。

完整证据位于：

- `reports/b3_real_v6k_3k_heldout_validity_audit.json`：原始结果认证失败项；
- `reports/b3_real_v6k_3k_forensic_replay/forensic_replay.json`：离线复算摘要；
- `reports/b3_real_v6k_3k_forensic_replay/metrics.replayed.json`：当前 schema 的逐样本
  指标；
- 同目录 `predictions.replayed.jsonl` 与 `references.posthoc_subset.jsonl`：复算输入。

## 3. 新代码如何防止同类结果再次出现

### 3.1 唯一 parser

训练后推理和 forensic replay 现在都调用
`sceneledger.eval.parser.parse_caption_output`。XML/atomic 解析不再在不同 CLI 中各自
复制，避免同一 raw text 因入口不同而产生不同的 format 状态。

### 3.2 `sceneledger-metrics-v2`

`sceneledger-evaluate` 生成的 metrics 必须包含：

- 每条样本的 event、caption、100 ms segment、边界、source count、pointer、
  hallucination、omission 和 strict-format 指标；
- 完整 raw parser status，不能从已经修复/解析后的 Ledger 倒推格式成功；
- `macro_caption_token_f1`，禁止只有 type+time 的 event-F1；
- 可从逐样本行精确重算的所有 corpus aggregate。

`validate_metrics_artifact` 会逐项重算并检查 schema、样本唯一性、格式证据完整性和
aggregate 一致性。legacy JSON 仍可用于取证，但不能通过论文结果认证。

### 3.3 结果认证新增检查

`sceneledger-audit-result` 现在额外要求：

1. metrics 与 inference report 的逐样本 strict-format 状态及总体 rate 完全一致；
2. metrics 符合当前语义指标 schema；
3. 原有的完整 test coverage、sample/source disjoint、冻结合同、制品哈希和人工审核
   门禁仍全部通过。

认证失败时输出 `claim_scope=diagnostic_only`；只有全部通过才输出
`claim_scope=paper_eligible` 和 `status=certified_generalization`。

### 3.4 无 GPU 复算旧结果

在仓库根目录安装项目后可复现实次取证：

```bash
sceneledger-forensic-replay \
  --train-config configs/model/b3_real_v6k_3k.yaml \
  --manifest data/derived/real_mix_v6_1k/manifest_compat.jsonl \
  --inference-report reports/b3_real_v6k_3k_heldout_infer_report.json \
  --original-metrics reports/b3_real_v6k_3k_heldout_metrics.json \
  --repo-root . \
  --output-dir reports/b3_real_v6k_3k_forensic_replay
```

该命令不读取音频、不加载 MOSS，也不需要 GPU。它会：

1. 从训练 YAML 重建 702 条真实 training members；
2. 用当前 parser 重解析 100 条已保存的 `raw_text`；
3. 从原 manifest 取回 post-hoc references；
4. 生成 metrics-v2，并分别复算 seen/unseen-by-ID；
5. 永久标记 `paper_eligible=false`，因为事后复算无法补回冻结 test 和 source identity。

## 4. 下一项实验是什么

下一步不需要再训练 v6k，也不需要立刻增加 loss、agent 或 RL。唯一有判定力的实验仍是
`docs/33_real_complex_three_fold_anchor.md` 的 120/120/120 六源三折锚点：

```text
可追溯 source catalogs
  -> source-group train/val/test 三折冻结
  -> 各折独立 recipe + scene plan
  -> mixture/stem/hash/复杂度自动门禁
  -> recipe 全审 + 60 条 test 试听
  -> 同一冻结 test 上 zero-shot
  -> 只用 train 折训练 B3
  -> 同一冻结 test 上 tuned
  -> metrics-v2 + result certification + 盲法语义评审
```

该锚点先回答三个基础问题：

1. 当前 mixer 是否能产出自然、可听、确实复杂且标签正确的数据；
2. 当前基线在真正 source-disjoint 的复杂音频上有多差；
3. SFT 是否同时改善 caption、时间、source count 和 pointer，而非只学会标签与时间模板。

只有这三点可重复成立，才值得扩大 train 到 1,000/5,000，并继续比较 LLM recipe、
recaption、显式 expert/agent 或 Track–Event 模型。否则同时改模型和数据仍无法归因。

## 5. 服务器执行者现在应做什么

按顺序执行，任何一步失败就停止：

1. 按 `docs/33` 准备 LibriSpeech、ESC-50、FSD50K catalog，并完成人工 source audit；
2. 生成并 100% 审核 train/val/test recipe review；
3. 渲染 mixture/stem，运行 data gate；
4. 试听 60 条冻结 test mixture 并生成通过的 human-audit summary；
5. 仅在上述证据全部通过后运行：

```bash
bash scripts/run_b3_real_complex_anchor.sh "$EXP_ROOT" "$MOSS_WEIGHTS" 1000
```

最终回传 `gate/`、三折 manifest、human audit summary、zero-shot/tuned inference 与
metrics、comparison、`validity_audit.json` 和 blinded review summary。若
`validity_audit.json` 不是 `certified_generalization`，结果只能用于定位故障，不能写入
论文主表。
