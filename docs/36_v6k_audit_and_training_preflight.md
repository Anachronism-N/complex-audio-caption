# v6k 审计与训练前置授权门禁

更新日期：2026-08-15

## 1. 结论

远端新增的 `real_mix_v6_1k` 把旧方案扩到了 1,000 条，但没有修复决定实验有效性的
数据问题，因此当前 **不要运行** `configs/model/b3_real_v6k_3k.yaml`：

- 每条只有 2 或 3 个 source/event，严格复杂 scene 比例为 0；
- 没有多说话人、speech+music+SFX 或 music+vocal 样本；
- 2,740 个 source 全部只有 `real:<role>` 占位路径，没有原始 recording/speaker/song
  identity 或 `source_group`；
- 1,000 条均没有 mixture hash 和 stem map，无法重放或做 source-level 归因；
- 没有冻结 train/val/test contract、数据质量 gate 或 mixture 人工审核；
- 旧训练器产生的 validation 不是预期的 100 条随机样本，而是 298 条完整模板组。

新增训练前置授权门禁会在加载模型和占用 GPU 之前拒绝该配置。下一步实验仍是
`docs/33_real_complex_three_fold_anchor.md` 中的七 stem/六 track、三折、人工审核
anchor；不是继续
扩大 v6k，也不是先调 loss 或训练步数。

## 2. 拉取内容的确定性审计

本轮拉取的远端主线提交为 `75a700d`，新增了 1,000 条 v6k manifest 和 3,000-step
训练配置。对兼容 manifest 运行 `diagnostic_v1` 得到：

| 指标 | v6k 实测 | 要求 | 结论 |
|---|---:|---:|---|
| scene 数 | 1,000 | >= 50 | 通过 |
| source/event 均值 | 2.74 / 2.74 | >= 3 / >= 3 | 失败 |
| 2-source / 3-source scene | 260 / 740 | — | 上限仍为 3 |
| overlap ratio 均值 | 0.496 | >= 0.15 | 通过 |
| strict-complex 比例 | 0.000 | >= 0.30 | 失败 |
| block-like 比例 | 0.260 | <= 0.30 | 通过但接近上限 |
| 多 voice 比例 | 0.000 | 目标任务需要 | 缺失 |
| speech+music+SFX 比例 | 0.000 | 目标任务需要 | 缺失 |
| music+vocal 比例 | 0.000 | 目标任务需要 | 缺失 |
| provenance 完整率 | 0.000 | 1.000 | 失败 |

因此 v6k 相比 200 条 v6 的主要变化是样本数，而不是任务覆盖。高 overlap 只能证明
2–3 条长块有重叠，不能证明模型会处理多说话人、密集 SFX、音乐人声或复杂事件转移。
冻结报告见 `reports/real_mix_v6k_complexity_audit.json`。

### 2.1 旧内部划分实际做了什么

配置写的是 `group_key: source_id`，但旧 trainer 实际把每条 scene 中排序后的 source
`path` 拼接并哈希。由于兼容转换只保留 4 个角色占位路径，1,000 条只形成 6 个路径
组合组。固定 seed 后，一个 298 条的组被整体选为 validation：

```text
validation = restaurant_busy 150 + concert_outdoor 148 = 298
training   = 其余五类模板 = 702
```

这既不是 raw-source 隔离，也不是同分布的 90/10 划分；训练集完全不含上述两个模板。
因此即使未来得到 v6k 的训练指标，也不能与三折 anchor 直接比较。

## 3. 新增代码：CPU-only training preflight

所有 `sceneledger.cli.train` 运行现在都会先执行 CPU 预检，再 import/load MOSS 权重。正式
训练必须同时满足：

1. manifest 非空、sample ID 非空且唯一，实际训练成员非空；
2. source 有可审核的原始路径或 `source_group`，不能只有 `real:<role>`；
3. 每条 mixture 都有 waveform hash 和 stem map；
4. `pre_split=true` 且 `expected_split=train`；
5. split contract 精确绑定当前 train manifest 的 hash 和 sample IDs；
6. 完整 experiment-data gate 已通过且所有制品 hash 未变化；
7. 完成的 human-audit summary 与相同 `dataset_id` 绑定；
8. 训练配置中的 `experiment_contract.dataset_id` 与以上证据一致。

预检会输出：实际被 trainer 访问的样本数、内部 validation 数、两侧模板分布、source
identity/hash/stem 统计、契约制品哈希和所有失败原因。正式 anchor runner 会在
zero-shot 和训练之前各验证一次，并冻结到：

```text
$EXP_ROOT/gate/training_preflight.json
```

这消除了“训练完成后才发现数据没准备好”的执行顺序错误。

## 4. 如何独立检查配置

安装更新后的项目后运行：

```bash
sceneledger-train-preflight \
  --config configs/model/b3_real_v6k_3k.yaml \
  --repo-root . \
  --output reports/real_mix_v6k_training_preflight.json
```

该命令对 v6k 应以非零状态退出，并报告：

```text
status = rejected_uncontracted_or_invalid
n_samples = 1000
n_actual_training_samples = 702
n_internal_validation_samples = 298
n_placeholder_sources = 2740
n_nonempty_mixture_hashes = 0
n_nonempty_stem_maps = 0
```

只有为了复现旧故障、且结果明确不用于论文时，才允许显式覆盖：

```bash
python -m sceneledger.cli.train \
  --config configs/model/b3_real_v6k_3k.yaml \
  --allow-exploratory-uncontracted \
  --preflight-report outputs/v6k_forensics/training_preflight.json
```

此时报告永久标记为 `exploratory_uncontracted`、`publication_eligible=false`。如果配置
声称使用正式契约但契约不完整或已损坏，即使带覆盖参数也会拒绝，防止把正式实验悄悄
降级为 legacy run。

## 5. 服务器下一步只做这一条链路

### Phase A：数据与人工门禁，不使用 GPU

严格执行 `docs/33_real_complex_three_fold_anchor.md` 第 3–5 节：

1. 准备并人工审核 LibriSpeech、ESC-50、FSD50K source catalog；
2. 生成 train/val/test 各 120 条七 stem、七 event、六 track scene specification；
3. 100% 审核三折 rule recipe；
4. 渲染 mixture 和 stems；
5. 通过 replay、stem-sum、active-RMS、复杂度、source-disjoint、multi-event-track
   和 stem temporal evidence contract；
6. 试听并完成 60 条冻结 test mixture audit；
7. 运行 summary，得到 `human_audit_summary.json: pass=true`。

任一步失败就停在对应的数据环节修复，不启动模型实验。

### Phase B：唯一获准的 GPU anchor

Phase A 全部通过后：

```bash
export MOSS_WEIGHTS=/path/to/moss_weights
bash scripts/run_b3_real_complex_anchor.sh "$EXP_ROOT" "$MOSS_WEIGHTS" 1000
```

脚本顺序是：生成绑定契约的 runtime config → 首次训练预检 → zero-shot 完整 test →
训练入口再次预检 → 只训练 frozen train → tuned 完整 test → 结果认证 → 盲法 A/B
人工评审。首次预检确保坏数据不会先浪费 zero-shot GPU；二次预检防止运行期间契约或
manifest 发生变化。

### Phase C：何时扩大规模

只有 120/120/120 anchor 同时满足以下条件，才把 **train** 扩到 1,000；val/test 保持
冻结，不得随着训练规模一起重采样：

- data gate、human audit、training preflight 和 result validity 全通过；
- tuned 相比同 test 的 zero-shot 在 caption、时间、source count、pointer 上有一致改善；
- 盲法双 reviewer 没有发现系统性 speaker confusion、SFX 替换或模板化幻觉；
- 改善不是只来自格式成功率或 event type/time 的宽松匹配。

如果 anchor 不通过，下一轮修改必须由失败类别决定：不可听就改 active-RMS/ducking，
组合不自然就改 recipe，时间错位就查 renderer/target，语义错误才讨论 source caption 或
模型。禁止在原因未定位时同时改数据、loss、模型和推理策略。

## 6. 本轮代码验证范围

单元测试覆盖四个关键分支：v6k 默认拒绝、legacy 显式覆盖仍为非论文状态、完整契约
放行、部分/损坏契约不可覆盖。正式 anchor runner 还会把训练预检报告持久化，供后续
result certification 和论文实验记录使用。
