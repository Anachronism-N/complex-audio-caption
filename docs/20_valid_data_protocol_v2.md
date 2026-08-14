# B3-complex-v2：下一步有效实验执行文档

## 1. 本轮只回答一个问题

在继续训练模型前，先确认我们能稳定生成一套满足以下条件的数据：

1. train、validation、test 的原始干声源完全不重叠；
2. test 不会因为脚本默认行为包含训练样本；
3. 主数据分布以多事件复杂场景为主；
4. 不再出现大量十几秒音频只有一个不足一秒音效的情况；
5. repeated-event 确实包含至少两个时间片段；
6. overlapping-speakers 确实发生时间重叠；
7. 每个结果都绑定 manifest 哈希、split 合同和 dataset ID。

这一阶段不比较新模型，不调 CARC 概率，也不根据模型指标修改数据阈值。只有数据门禁通过，才允许启动 GPU 训练。

## 2. 为什么旧 B3 数据不能继续使用

对 `data/derived/b3_5k/manifest.jsonl` 的复核结果为：

- 单事件场景占 13.78%；
- 活动覆盖率低于 20% 的场景占 14.16%；
- 活动覆盖率低于 50% 的场景占 44.74%；
- 尾部静音超过 10 秒的场景占 45.08%；
- `isolated_sfx` 的平均活动覆盖率只有 4.0%；
- `isolated_sfx` 的平均尾部静音为 15.7 秒。

此外，旧推理命令直接读取完整 manifest。CARC 的 500 条评测中有 450 条属于训练折，训练集污染率为 90%。因此旧结果只能归档为开发诊断，不能进入论文主表。

可以用新门禁复核任意单个旧 manifest；该命令预期以非零状态退出并写出失败报告：

```bash
python -m sceneledger.cli.audit_mixtures \
  --manifest data/derived/b3_5k/manifest.jsonl \
  --quality-config configs/data/mixture_quality.yaml \
  --profile release \
  --output reports/b3_5k_mixture_quality.json
```

## 3. v2 代码做了什么

### 3.1 显式的声源隔离 fold

新配置不再从一个混合后的 manifest 临时切分：

- train 使用 synthetic source index `000–799`；
- validation 使用 `800–899`；
- test 使用 `900–999`。

三个 fold 独立渲染，scene ID、采样 seed 和 template seed 也相互独立。真实数据阶段应把这里的 index range 替换为按原始 recording/source group 预先划分的三个 source catalog，不能先混音再随机拆 scene。

### 3.2 新场景生成规则

新规则只在 `b3_complex_v2_*.yaml` 中显式启用；旧 YAML 的采样语义保持不变。

- music 和 ambience 通过短 crossfade 循环并裁剪到场景长度；
- 前景事件可以分布到更完整的时间轴；
- `isolated_sfx` 时长缩短到 2–5 秒；
- `isolated_sfx` 权重降至约 3%，只承担稀疏诊断作用；
- `repeated_event` 的 SFX 至少重复两次，并加入持续 ambience；
- repeated SFX 的实例在时间轴上拉开，而不是全部挤在开头；
- overlapping speakers 的 onset 被约束到邻近位置；
- source ID 使用稳定且类型不冲突的命名，避免 speech 与 SFX 同为 `Sxx`。

旧 manifest 没有 `loop_to_scene` 时默认为 `false`，因此仍可按旧音频语义重放。

### 3.3 数据门禁

`sceneledger.cli.validate_experiment_data` 一次生成：

- `split_contract.json`；
- `train_mixture_quality.json`；
- `val_mixture_quality.json`；
- `test_mixture_quality.json`；
- 三个 fold 的 canonical references；
- `experiment_data_summary.json`。

在此之前，`sceneledger.cli.preflight_data` 只采样 scene graph、不渲染 WAV，验证三折复杂度目标并冻结 `scene_plan_sha256`。最终 data summary 必须绑定通过的 `scene_plan_preflight.json`，且 manifest 中的 scene plan 哈希必须完全一致。

release profile 的初始硬阈值为：

| 检查 | 阈值 |
|---|---:|
| 单事件场景比例 | ≤ 5% |
| 活动覆盖率低于 30% 的场景比例 | ≤ 10% |
| 尾部静音超过 5 秒的场景比例 | ≤ 10% |
| 任意连续静音超过 5 秒的场景比例 | ≤ 10% |
| sparse template 比例 | ≤ 5% |
| 重复事件少于两个 SFX spans | 0% |
| 多说话人重叠率低于 10% 的比例 | ≤ 10% |
| 场景内重复底层 source path 比例 | 0% |
| 平均 source count | ≥ 3.4 |
| simple / medium / complex 比例 | 15–30% / 40–60% / 20–40% |
| 三个新增复杂模板各自比例 | ≥ 8% |
| complex 样本低 overlap 比例 | ≤ 20% |
| 重复 source ID | 0% |

这些是首版工程验收阈值，不是 TAC 论文中的既定阈值。任何阈值调整必须新增 profile 名称，不能覆盖已有 profile 后重跑。

## 4. 环境检查

在服务器仓库根目录执行：

```bash
git rev-parse HEAD
python --version
ffmpeg -version | head -n 1
python - <<'PY'
import numpy, scipy, soundfile, yaml
print("data dependencies: OK")
PY
```

如果仓库没有以 editable 模式安装：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

先运行 CPU 测试：

```bash
pytest -q tests/unit/test_experiment_data.py
```

预期新增测试全部通过。测试失败时不要生成完整数据。

## 5. 完整生成命令

推荐在容量充足的 CephFS 目录执行。脚本第一个参数是输出根目录：

```bash
bash scripts/run_b3_complex_v2_data.sh \
  /path/on/cephfs/b3_complex_v2 \
  2>&1 | tee /path/on/cephfs/b3_complex_v2_run.log
```

脚本依次执行：

1. 渲染并 replay-validate 4000 条 train；
2. 渲染并 replay-validate 500 条 validation；
3. 渲染并 replay-validate 500 条 test；
4. 检查三个 fold 的 sample ID 与原始 source identity 泄漏；
5. 对每个 fold 单独执行分布门禁；
6. 导出 test references 和最终 summary。

如果需要指定 Python：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
  bash scripts/run_b3_complex_v2_data.sh /path/on/cephfs/b3_complex_v2
```

## 6. 必须检查的产物

首先检查最终状态：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/path/on/cephfs/b3_complex_v2")
summary = json.loads((root / "gate/experiment_data_summary.json").read_text())
print("pass:", summary["pass"])
print("dataset_id:", summary["dataset_id"])
print("failed_checks:", summary["failed_checks"])
PY
```

只有同时满足以下条件才进入人工试听：

- `pass` 为 `true`；
- `failed_checks` 为空；
- train/val/test 三份 quality report 均为 `passed`；
- split contract 中三个 source-disjoint 检查均通过；
- dataset ID 非空并记录到实验日志。

如果失败，查看对应 quality report 的 `violation_samples`。该字段最多保留 200 个代表样本及失败原因，便于直接试听，不要通过放宽阈值掩盖问题。

## 7. 人工试听协议

自动门禁通过后，从每个 fold 分层抽取：

- 每个 template 至少 5 条；
- active ratio 最低的 10 条；
- trailing silence 最长的 10 条；
- overlap ratio 最低的 10 条 overlapping-speakers；
- SNR 最低、T60 最大、带 echo 的场景各 10 条。

每条记录：

- 音频是否损坏或削波；
- caption 中每个事件是否可听见；
- 是否存在未标注的明显事件；
- onset/offset 是否在约 0.1 秒容差内合理；
- speech、lyrics、music、SFX 类型是否正确；
- 重叠、混响、回声是否符合 metadata；
- 是否仍有不自然的长静音或明显循环接缝。

建议至少两名标注者独立审核 test fold。任何系统性问题都返回生成器修复，不能在模型训练后再解释。

## 8. 数据通过后才运行的基线

将生成目录映射或链接到：

```text
data/derived/b3_complex_v2/
```

然后执行受 split contract 保护的基线：

```bash
python -m sceneledger.cli.train \
  --config configs/model/b3_slot_aware_valid_v2.yaml
```

训练入口会验证 train manifest 的 SHA-256 是否与 `split_contract.json` 一致，并禁止对已经冻结的 train manifest再次内部切分。

只能在 test manifest 上推理：

```bash
python -m sceneledger.cli.infer \
  --manifest data/derived/b3_complex_v2/test/manifest.jsonl \
  --audio-base data/derived/b3_complex_v2/test \
  --backend moss \
  --model-path /tmp/moss_weights \
  --lora-path outputs/b3_slot_aware_valid_v2/lora \
  --greedy \
  --include-lyrics \
  --split-contract data/derived/b3_complex_v2/gate/split_contract.json \
  --data-gate-summary data/derived/b3_complex_v2/gate/experiment_data_summary.json \
  --expected-split test \
  --output reports/b3_valid_v2_predictions.jsonl \
  --report reports/b3_valid_v2_infer_report.json
```

评测同样必须绑定 test split：

```bash
python -m sceneledger.cli.evaluate \
  --prediction reports/b3_valid_v2_predictions.jsonl \
  --reference data/derived/b3_complex_v2/gate/test_references.jsonl \
  --split-contract data/derived/b3_complex_v2/gate/split_contract.json \
  --data-gate-summary data/derived/b3_complex_v2/gate/experiment_data_summary.json \
  --expected-split test \
  --inference-report reports/b3_valid_v2_infer_report.json \
  --output reports/b3_valid_v2_metrics.json \
  --pretty
```

当 split contract 生效时，`--limit` 被禁止，且推理和评测都强制保存/读取 inference report。预测、reference 或 inference report 只要缺少、增加、重复一个 sample ID，或者 prediction 的 SHA-256、dataset ID、split 任一不匹配，评测都会直接失败。格式成功率只来自逐样本原始输出解析状态；不能因为 tolerant parser 产出了一个空 Ledger 就默认成功。

## 9. 明确的停止条件

出现以下任何情况都停止 GPU 实验：

- data summary 未通过；
- source leakage 非零；
- 任一 manifest 在生成 contract 后发生变化；
- 人工审核发现系统性标签错误、长静音或循环伪影；
- test predictions 的 sample ID 与冻结 test fold 不完全一致；
- 使用了旧 `b3_unified` 或 `b3_5k` 全 manifest 评测命令。

## 10. 与真实数据流程的关系

`b3_complex_v2` 是验证“采样—渲染—切分—审计—评测”闭环的 synthetic anchor，不是最终论文训练数据。下一阶段真实 D0 必须提供：

- speech、vocal、music、SFX、ambience 的真实 source catalog；
- 每条源音频的许可、原始 recording/source group、hash 和质量报告；
- vocal 的逐字歌词或明确排除；
- 在混音前完成 recording-level train/val/test 划分。

只有真实 D0 和本文件的数据门禁都通过，才开始比较 slot-aware、CARC、显式分轨或 agent 重写方案。CARC 还需要独立的删源挑战集与 paired consistency objective，不能仅凭原始混音上的 hallucination 数量宣称反事实学习有效。
