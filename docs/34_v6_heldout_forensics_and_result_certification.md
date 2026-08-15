# v6 held-out 结果取证与下一实验认证门禁

更新日期：2026-08-15

## 1. 远端最新结果的结论

远端提交 `db645ae` 报告 B3-real-v6-3k 在 `rv6_0181`--`rv6_0200`
上的 event-F1 为 0.967，并据此声称模型已经泛化。该结论不成立。

训练配置没有把前 180 条作为训练集。`sceneledger.cli.train` 实际读取完整
200 条 manifest，再按配置中的 `group_key=source_id` 和 `val_fraction=0.1`
调用 group split。v6 manifest 又没有保存真实原始文件身份，只留下
`real:speech`、`real:sfx`、`real:music`、`real:ambience` 等占位路径；训练器
最终按六种路径组合分组，实际训练 131 条、内部 validation 69 条。

因此，直接取 `manifest[180:]` 不等于 held-out：

| 子集 | 样本数 | event-F1 | onset MAE | hallucination | omission |
|---|---:|---:|---:|---:|---:|
| 报告的全部“held-out” | 20 | 0.967 | 0.178 s | 2 | 2 |
| 实际被训练使用 | 15 | 1.000 | 0.000 s | 0 | 0 |
| sample-ID 未被训练使用 | 5 | 0.867 | 0.710 s | 2 | 2 |

剩余 5 条也不能称为 source-disjoint test。v6 未保留 ESC-50/UrbanSound8K
文件、GTZAN track、speaker/utterance、音频哈希或 uploader 等原始身份，无法
证明训练和测试没有复用同一干声；构造脚本中的 speech bank 本身只有两段音频。
这 5 条只能作为诊断样本。

此外，当前 event-F1 的事件匹配主要衡量 type 和时间 tIoU，不要求 caption
语义正确。该报告没有 `caption_token_f1`，所以即使切分正确，也不能用 0.967
证明细粒度 caption 能力。

完整机器报告：`reports/b3_real_v6_3k_heldout_validity_audit.json`。

## 2. 本轮新增的结果认证

新增命令 `sceneledger-audit-result`。它不是新的模型尝试，而是在提交论文结果
之前回答“这个数字能否解释”的 fail-closed 门禁。它会：

1. 从训练 YAML 重建训练器实际访问的 sample ID，而不是相信实验脚本中的
   `held_out` 注释；
2. 要求 metrics 与 inference report 的 ID 唯一、顺序一致，并完整覆盖冻结的
   eval manifest；
3. 检查训练和评测 sample ID 零重叠；
4. 使用 `source_group`、`leakage_groups` 或可追溯路径检查原始声源隔离，并拒绝
   `real:*` 这类占位身份；
5. 验证 split contract、experiment data gate、train/test manifest 哈希、
   metrics/inference 的 `dataset_id` 和 prediction hash 绑定；
6. 自动分别输出 seen/unseen 子集指标，防止训练样本的完美分数掩盖真正未见
   样本的退化。

只有 `status=certified_generalization` 才能汇入论文主表。缺合同、只评前 N 条、
source identity 不可审计或任一 overlap 都得到
`status=invalid_generalization_claim`。`--require-pass` 会以退出码 2 停止实验。

复核旧结果的命令：

```bash
sceneledger-audit-result \
  --train-config configs/model/b3_real_v6_3k.yaml \
  --eval-manifest data/derived/real_mix_v6/manifest_compat.jsonl \
  --metrics reports/b3_real_v6_3k_heldout_metrics.json \
  --inference-report reports/b3_real_v6_3k_heldout_infer_report.json \
  --repo-root . \
  --output reports/b3_real_v6_3k_heldout_validity_audit.json
```

旧 `scripts/eval_heldout.py` 现在只保留为失败保护，避免再次用
`manifest[180:]` 产生同类结论。

## 3. 下一步实验没有改变：先完成真实三折数据锚点

不要扩大 v6、不要继续调 v6 训练步数，也不要对这 20 条做新的模型消融。
下一项可判定实验仍是 `docs/33_real_complex_three_fold_anchor.md` 中的
Real-Complex 三折锚点：

```text
LibriSpeech / ESC-50 / FSD50K source catalog
  -> speaker / recording / uploader 三折隔离
  -> train / val / test 独立 scene plan 与 recipe review
  -> 波形、复杂度、stem、split contract 自动门禁
  -> 60 条 test mixture 人工听审
  -> 完整 test zero-shot
  -> 仅 train 折训练 B3
  -> 同一完整 test tuned
  -> result certification
```

数据和人工审核全部通过后，服务器只需运行：

```bash
bash scripts/run_b3_real_complex_anchor.sh "$EXP_ROOT" "$MOSS_WEIGHTS" 1000
```

该脚本现在会在最后生成：

```text
$EXP_ROOT/evaluation/b3_tuned/validity_audit.json
```

若认证失败，脚本会停止，`comparison.json` 只能用于排错。认证通过后仍必须同时
查看 event-F1、caption token-F1、100 ms tolerance、source-count、pointer、
hallucination 和 omission；不能再次只汇报 event-F1。

## 4. 本轮需要执行者做什么

当前不需要再运行 v6。按 `docs/33` 完成三件事：

1. 准备三个 source catalog，并将每个数据集的 source audit 扩到
   train/val/test 每类各至少 10 条；
2. 生成 train/val/test recipe review CSV，人工填写全部 plausibility 和
   label-compatible 字段；
3. 渲染通过后试听并填写 60 条 test mixture audit，再启动上述 B3 脚本。

需要回传的最低结果是 `gate/`、三折 manifest、human audit summary、
zero-shot/tuned metrics、comparison 和 `validity_audit.json`。缺少最后一个文件
时，不再接受“held-out/generalizes”作为实验结论。
