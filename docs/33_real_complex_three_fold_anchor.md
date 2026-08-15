# 下一步实验：Real-Complex 三折数据锚点与无泄漏 B3

更新日期：2026-08-16

## 1. 本轮只回答什么

旧 `real_mix_v6` 的 3,000-step 报告不能作为泛化结论：50 条评测中有
38 条属于训练集合，而且 event-F1 的有效匹配只要求类型相同、时间 tIoU
不低于 0.3，并不要求 caption 语义正确。

本轮只回答一个更小但有效的问题：

> 在 LibriSpeech、ESC-50、FSD50K 的原始 speaker/recording/uploader group
> 严格三折隔离后，B3 在 120 条六源 train scene 上训练，能否在完整的
> 120 条冻结 test scene 上同时改善事件、caption、时间和 track 指标？

这里的每条 scene 固定包含两个不同说话人、一条 ambience 和三条前中后
分布的 SFX。默认 train/val/test 各 120 条。这不是论文最终规模，而是验证
数据、训练和评测契约的第一个无泄漏锚点；它通过后才能把 train 扩到 1k。

## 2. 新代码解决了哪些旧问题

1. `make_complex_speech_sfx_experiment.py` 从 prepared catalog 的
   `train.jsonl`、`val.jsonl`、`test.jsonl` 分别构造配置，禁止再从一个
   200 条 manifest 内部临时切分。
2. 三折分别冻结 inventory、rule recipe、seed 和 source split；每个 recipe
   必须在渲染前完成 plausibility/label-compatible 审核。
3. `validate_experiment_data` 现在把每一折的 complexity report、manifest
   SHA-256、mixture-quality report 和 frozen references 一起写入数据契约。
   任何文件改变后，训练和评测入口都会拒绝继续。
4. 训练配置只能在 60 条 test mixture 人工审核通过后生成。
5. 推理入口禁止 `--limit`，先在完整冻结 test 上运行 zero-shot，再训练
   B3，并在完全相同的 test 上运行 tuned arm。
6. 评测新增 `macro_caption_token_f1`。event-F1 现在明确称为类型/时间指标；
   caption omission 按零分计入 lexical token-F1，不能再用高 event-F1
   代替语义正确率。

`macro_caption_token_f1` 只是可重复的词汇诊断，不处理同义改写。论文阶段
仍需人工 semantic precision/unsupported-attribute 评测，但它至少能立即暴露
“类型和时间正确、caption 内容完全错误”的情况。

## 3. 前置条件：重新审核三个 source split

三个 source audit report 必须显式包含：

```json
"required_splits": ["train", "val", "test"]
```

每个 required split、每个 required kind 至少要有 10 条人工审核任务。source
audit sheet 按 train/val/test round-robin 采样，因此 preparer 必须使用
`--audit-per-kind 30`。旧的 ESC-50 10-row audit 不能复用。

建议建立新目录，避免覆盖已经用于旧报告的制品：

```bash
export REPO=/path/to/complex-audio-caption
export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/real_complex_anchor_v1
export LIBRI_ROOT="$DATA_ROOT/librispeech_full_clean"
export ESC_ROOT="$DATA_ROOT/esc50/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
export FSD_ROOT="$DATA_ROOT/fsd50k/FSD50K"

cd "$REPO"
python -m pip install -e '.[data,dev,moss]'
```

LibriSpeech 必须包含足够的 speaker group，使三个 split 都能满足复杂度门禁；
推荐使用之前下载脚本的 `full-clean` profile。分别运行当前 preparer，三个
preparer 都使用 `--audit-per-kind 30`：

```bash
sceneledger-prepare-librispeech \
  --root "$LIBRI_ROOT" \
  --output-dir "$RUN_ROOT/sources/librispeech" \
  --audit-per-kind 30

sceneledger-prepare-esc50 \
  --metadata "$ESC_ROOT/meta/esc50.csv" \
  --audio-root "$ESC_ROOT/audio" \
  --output-dir "$RUN_ROOT/sources/esc50" \
  --audit-per-kind 30

sceneledger-prepare-fsd50k \
  --root "$FSD_ROOT" \
  --output-dir "$RUN_ROOT/sources/fsd50k" \
  --allow-license CC0-1.0 \
  --allow-license 'CC BY 3.0' \
  --audit-per-kind 30 \
  --min-per-kind-per-split 50
```

逐条试听三个 `prepared/source_audit.csv`，填写
`audible_y_n`、`caption_correct_y_n`、`kind_correct_y_n`。随后对每个数据集
运行下面的验证；将路径中的 `DATASET` 分别替换为三个目录：

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$RUN_ROOT/sources/DATASET/prepared/source_catalog_report.json" \
  --audit-csv "$RUN_ROOT/sources/DATASET/prepared/source_audit.csv" \
  --output "$RUN_ROOT/sources/DATASET/prepared/source_audit_report.json" \
  --min-per-kind 30 \
  --min-pass-rate 0.90 \
  --required-split train \
  --required-split val \
  --required-split test \
  --min-per-kind-per-required-split 10
```

任一审核失败时修 source bank，不生成 mixture。

## 4. 冻结三折实验 specification

```bash
export SPEC_ROOT="$RUN_ROOT/spec"
export EXP_ROOT="$RUN_ROOT/experiment"

python scripts/make_complex_speech_sfx_experiment.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$RUN_ROOT/sources/librispeech/prepared" \
  --esc50-audio-root "$ESC_ROOT/audio" \
  --esc50-prepared "$RUN_ROOT/sources/esc50/prepared" \
  --fsd50k-root "$FSD_ROOT" \
  --fsd50k-prepared "$RUN_ROOT/sources/fsd50k/prepared" \
  --train-count 120 \
  --val-count 120 \
  --test-count 120 \
  --output-dir "$SPEC_ROOT"
```

这一步只生成配置，不渲染音频。检查并填写：

```text
$SPEC_ROOT/train.rule_recipe_review.csv
$SPEC_ROOT/val.rule_recipe_review.csv
$SPEC_ROOT/test.rule_recipe_review.csv
```

每行都必须填写 `plausible_y_n` 和 `label_compatible_y_n`。如果某个组合不合理，
标 `n` 并修正 recipe/source 规则后重新生成整个新 specification；不要把被拒绝
recipe 留在数据中，也不要手工修改 source ID 或时间戳。

## 5. 渲染、三折门禁和 mixture 人工审核

```bash
bash scripts/run_complex_speech_sfx_experiment.sh "$SPEC_ROOT" "$EXP_ROOT"
```

脚本严格按以下顺序运行：

```text
三折 recipe review 100%
  -> scene-plan preflight
  -> train/val/test render + replay/stem-sum
  -> source identity split contract
  -> 三折 waveform/complexity audit
  -> frozen references
  -> 60 条 test mixture 人工任务
```

自动门禁通过后，完成：

```text
$EXP_ROOT/gate/human_audit_tasks.csv
```

必须逐条检查两个 speaker、三个 SFX、ambience、时间对齐、重叠、caption、
stem-sum 和整体自然度。完成后运行：

```bash
bash scripts/summarize_complex_speech_sfx_experiment.sh "$EXP_ROOT"
```

GO 标准是 severe=0、60 条中 overall failure 不超过 6，且同一失败 criterion
不超过 6。失败时不启动 GPU 训练。

## 6. Zero-shot 与 B3 的同 test 对比

```bash
export MOSS_WEIGHTS=/path/to/moss_weights
bash scripts/run_b3_real_complex_anchor.sh "$EXP_ROOT" "$MOSS_WEIGHTS" 1000
```

该命令会：

1. 验证 data gate、split contract、complexity report 和 human summary 的哈希；
2. 在完整 120 条冻结 test 上运行原始 MOSS zero-shot；
3. 只使用 frozen train manifest 训练 1,000 个 micro-steps；
4. 在同一个完整 test 上运行 B3 tuned；
5. 输出 `$EXP_ROOT/evaluation/comparison.json`。
6. 用训练配置、完整 test manifest、指标、推理报告及数据合同生成
   `$EXP_ROOT/evaluation/b3_tuned/validity_audit.json`；认证失败时以非零退出。
7. 认证通过后生成 60 条 zero-shot vs tuned 随机盲法 A/B 人工语义评审任务。

推理没有 `--limit`，test ID 必须与 contract 完全一致。任何前 50 条评测、
train manifest 推理或修改 references 的结果都不会通过入口校验。
此外，只有 `validity_audit.json` 的 `status=certified_generalization` 才能作为
论文 held-out 结果；详细门禁见 `docs/34_v6_heldout_forensics_and_result_certification.md`。
盲法模型评审的填写和揭盲流程见 `docs/35_blinded_model_semantic_review.md`。

## 7. 如何判断结果

同时查看以下指标，不能只看 event-F1：

| 指标 | 回答的问题 |
|---|---|
| strict format | 输出是否满足结构语法 |
| event P/R/F1 | 类型和时间 tIoU 是否形成有效事件匹配 |
| caption token-F1 | 与 source transcript/保守类别描述的词汇一致性 |
| onset/offset MAE、±0.1s | 100 ms 网格是否真的对应边界精度 |
| hallucination/omission | 是否以保守输出换取表面准确率 |
| source-count MAE | 六个 source 是否被恢复 |
| pointer accuracy | event 是否归属正确 track |

本轮模型 GO 不预注册某个绝对高分，而要求 tuned 相比 zero-shot 至少满足：

- event recall、caption token-F1、±0.1s 命中率均不下降；
- hallucination 和 omission 不出现一升一降的明显失衡；
- source-count MAE 下降；
- pointer accuracy 上升；
- 失败样本人工检查没有系统性说话人混淆或 SFX 语义替换。

如果 event-F1 上升但 caption token-F1、source count 或 pointer 不改善，结论只能是
“学会了六槽格式/时间模板”，不能声称复杂 caption 能力提升。只有这一锚点成立后，
才把 train count 扩到 1,000，并保持 val/test、split contract 和人工 test 不变。

## 8. 需要回传的文件

不要上传受许可限制的原始音频。请回传：

- 三个 source catalog/preparation/audit report 与完成的 audit CSV；
- `$SPEC_ROOT/experiment_spec.json` 和三折 recipe review report；
- `$EXP_ROOT/scene_plan_preflight.json`；
- `$EXP_ROOT/gate/experiment_data_summary.json`、split contract、三折 quality/
  complexity report、human audit summary；
- `$EXP_ROOT/evaluation/comparison.json`；
- zero-shot/tuned 的 metrics 和 inference report；
- 若失败，3--5 条代表性 mixture/stem 以及中文听感说明。
