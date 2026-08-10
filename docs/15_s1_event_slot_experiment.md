# S1a-valid：事件槽实验实现与服务器运行协议

本文档对应 `src/sceneledger/models/event_slots.py`、
`src/sceneledger/losses/set_prediction.py` 和
`src/sceneledger/cli/train_slots.py`。它把早期 S1a 原型收敛为一个可以复查数据划分、特征、
随机种子、checkpoint 和指标的实验。S1a 只回答一个问题：在相同 B3-valid 音频与冻结 MOSS
特征上，无序 event slots 能否稳定恢复事件类型、100 ms activity 和边界 envelope；它尚不生成
caption 文本，也不预测 track identity。

## 1. 前置门槛

实验严格按以下顺序运行：

1. 完成 TAG 2021 的 0--6 阶段，且 `runs/tag2021/reproduction_summary.json` 中
   `pass=true`；
2. 按 `docs/14_valid_experiment_pipeline.md` 构建 B3-valid；
3. 确认 `runs/b3_valid/sft/train_manifest.jsonl` 和 `val_manifest.jsonl` 存在；
4. 运行 S1a-valid。

`run_b1_official.sh`、`run_b3_valid.sh` 和 `run_s1_valid.sh` 默认都会调用
`scripts/repro/require_anchor_pass.py`。TAG 没通过时脚本退出，不会产生可误认为正式结果的后续
报告。若只是开发代码单元测试，应直接运行 `pytest`，不要伪造 `pass=true` 的锚点报告。

## 2. 相对早期 S1a 的修复

| 问题 | 当前实现 |
|---|---|
| 按 manifest 顺序切前 90%/后 10% | 使用 B3-valid 已冻结的 train/val manifest，再从 train 内划 source-disjoint calibration |
| 只设置 Python shuffle seed | 固定 Python、NumPy、Torch、CUDA，并启用 deterministic algorithms |
| eventness 正样本权重方向错误 | 使用 `(K-N_pos)/N_pos`，再执行可配置缩放与上限截断 |
| type/activity/boundary loss 随事件数线性增大 | 对全 batch 的 matched events 求均值 |
| 多段 SFX 被首尾区间填满 | target 和 prediction 都保留 disjoint activity spans |
| slot 聚合缺少显式时序位置 | 在 100 ms memory grid 上加入 learned temporal embedding |
| 在报告 validation 上 sweep 阈值 | 只在 calibration 上按 micro Event-F1 选阈值，再一次性评价 validation |
| `done.flag` 无法判断缓存身份 | cache manifest 绑定 manifest SHA-256、模型元数据、样本集合和存储 dtype |
| 注释称 fp16，实际存 fp32 | embedding 以 fp16 存储，训练加载时恢复 fp32 |
| 只在训练结束保存权重 | 定期验证，保存 `best.pt` 和 `last.pt`，支持严格 config-hash resume |
| 指标只写在 commit message | 保存 prediction、reference、完整 metrics、run manifest 和 summary |

Hungarian assignment 使用三项 soft cost：正确事件类型的负对数概率、activity Dice cost 和按
clip duration 归一化的 boundary L1 cost。assignment 本身不参与反向传播；匹配后再计算
eventness BCE、type CE、activity Dice 和按 clip duration 归一化的 boundary L1 loss。

### 2.1 对最新 S1a-v2 结果的解释

`2217194` 报告 threshold=0.40 时 onset MAE=0.008s，但该点 recall=0.010、F1=0.004；MAE
只在成功匹配的极少事件上计算。因此它说明“少数高置信匹配的边界可落在相邻 0.1s grid”，不能
说明模型整体达到 10ms 定位，也不能证明 boundary-only 优于 activity。新实现把 boundary head
并入同一模型，并同时输出：

- `matched_boundary_count` 与 `boundary_reference_coverage`；
- 按 matched event 加权的 onset/offset MAE；
- micro/macro Event-F1、SegF1、hallucination 与 omission；
- activity-only、boundary-only、hybrid 三种相同 checkpoint 解码结果。

历史 `scripts/train_s1v2.py` 和 `s1v2_threshold_sweep.py` 现为兼容入口，内部调用本协议，不再
读取过期 B3 synthetic manifest 或在最终 validation 上选择阈值。

最新 `scripts/train_s1v3_joint.py` 仍是 joint-encoder 探索入口，不是 primary S1 runner。旧提交
中的 gradient accumulation 会在每个 micro-step 清零梯度，warmup 也没有落在 optimizer update
上；其 F1=0.100 不能证明 differential LR 或 slot architecture 失败。当前入口已改为只读取通过
`data_reproduction_summary.json` gate 的 B3-valid train/val manifests，并修正 accumulation、
update-based warmup/cosine 与随机种子。它仍缺少本协议的 calibration checkpoint/threshold
选择，因此服务器数字只能记为 exploratory，不能和 primary S1 表直接合并。

历史 `configs/model/b3_permuted.yaml` 同样不能作为方法证据：训练器 shuffle 后，canonical
formatter 会再次按 onset/type/id 排序，500/500 个 target 都未变化；共用 RNG 只改变了后续 epoch
的样本顺序。训练入口现在会拒绝 `shuffle_events=true`，真实音源池通过前不再开展该消融。

## 3. 一次完整运行

服务器应先安装项目和 MOSS 依赖，并把 MOSS-Audio 放在仓库的 `third_party/MOSS-Audio`；
模型权重目录通过环境变量传入：

```bash
cd /path/to/complex-audio-caption
python -m pip install -e '.[dev,audio,moss]'

MODEL_DIR=/models/MOSS-Audio-4B-Instruct \
B3_WORK_DIR=/data/sceneledger/runs/b3_valid \
S1_WORK_DIR=/data/sceneledger/runs/s1_valid \
bash scripts/run_s1_valid.sh
```

脚本支持分阶段执行：

```bash
# 只抽取并审计特征；后续所有消融可复用该目录
STAGE=cache MODEL_DIR=/models/moss bash scripts/run_s1_valid.sh

# 从头训练
STAGE=train MODEL_DIR=/models/moss bash scripts/run_s1_valid.sh

# 从 last.pt 恢复；配置必须与 checkpoint 完全一致
STAGE=resume MODEL_DIR=/models/moss bash scripts/run_s1_valid.sh

# 只重新评测 best.pt
STAGE=evaluate MODEL_DIR=/models/moss bash scripts/run_s1_valid.sh
```

若移动了输出目录导致 config hash 变化，应把它视为新实验，不要强行加载旧 checkpoint。特征缓存
若因数据或模型变化失配，程序会要求显式传入 `--force-cache`；这用于防止新实验静默读取旧特征。

## 4. 关键配置

主配置是 `configs/model/s1_event_slots.yaml`：

- 24 个 event slots，对应方法设计中的容量；
- 4 层、8 heads、hidden size 768 的 Transformer decoder；
- 冻结 MOSS embedding，仅训练 event-slot head；
- 10000 steps，500-step warmup，每 500 steps 用 calibration loss 选择 checkpoint；
- eventness/type/activity/boundary loss 权重为 `1/1/2/1`；
- 从 B3-valid train 内划 10% source-disjoint calibration；
- eventness threshold 只在 calibration 候选集合上按 micro Event-F1 选择；
- activity threshold 固定为 0.5；
- evaluation tIoU gate 为 0.3。

命令行可以覆盖服务器路径，以及 `--steps`、`--seed`、`--n-slots`、
`--disable-temporal-embedding`、`--positive-weight-scale`、`--activity-weight`、
`--boundary-weight`、`--activity-cost-weight`、`--boundary-cost-weight` 和
`--primary-decode-mode`。所有覆盖后的值都会写入 `run_manifest.json`，
不会只存在于 shell history。

## 5. 输出契约

每个实验目录必须包含：

| 文件 | 用途 |
|---|---|
| `features/cache_manifest.json` | 特征缓存的数据、模型和样本身份 |
| `model/run_manifest.json` | 最终配置、git commit、运行时、随机种子和数据数量 |
| `model/split.json` | 实际 train/calibration/val sample IDs；三者 source leakage 必须为 0 |
| `model/best.pt` | 最低 calibration loss checkpoint |
| `model/last.pt` | 可恢复的最后 checkpoint |
| `model/val_predictions.jsonl` | S1a 事件预测，保留多个 spans |
| `model/val_references.jsonl` | 本次评测实际使用的 reference |
| `model/threshold_selection.json` | calibration 阈值候选、选择目标与最终阈值 |
| `model/val_metrics_{activity,boundary,hybrid}.json` | 三种解码的完整指标 |
| `model/val_metrics.json` | primary decode mode 的兼容指标入口 |
| `model/run_summary.json` | 便于汇总表读取的关键结果与阈值 |

正式记录结果时，以 `run_summary.json` 与 `run_manifest.json` 为准，不从终端日志手工抄数字。

## 6. 消融实验

以下命令共享同一份 MOSS feature cache：

```bash
MODEL_DIR=/models/MOSS-Audio-4B-Instruct \
B3_WORK_DIR=/data/sceneledger/runs/b3_valid \
ABLATION_ROOT=/data/sceneledger/runs/s1_ablation \
bash scripts/run_s1_ablation.sh
```

默认运行：`main`、`slots8`、`slots16`、`slots32`、`no_temporal_embedding`、
`no_positive_weight`、`activity_only`、`boundary_only`。也可以只运行指定子集：

```bash
bash scripts/run_s1_ablation.sh main slots8 no_temporal_embedding
```

比较时重点报告 Event-F1、SegF1@100ms、onset/offset MAE 及其 reference coverage、
hallucination、omission，并按 overlap ratio、source count 和事件类型分层。`activity_only` 与
`boundary_only` 判断两种时间头是否互补；两项消融同时把对应 loss 与 Hungarian cost 置零，避免
“未训练的 head 仍改变匹配结果”这一混杂变量。slot 数消融判断容量不足与过多 null slots 的权衡。

消融完成后生成统一 JSON/CSV 表：

```bash
python scripts/collect_s1_results.py /data/sceneledger/runs/s1_ablation \
  --output-json /data/sceneledger/reports/s1_ablation.json \
  --output-csv /data/sceneledger/reports/s1_ablation.csv
```

汇总器会拒绝缺少 `run_manifest.json`/`split.json`、记录了 source leakage，或 summary 与
manifest config hash 不一致的运行。

## 7. 与 B3 自回归基线的公平比较

S1a 与 B3-valid 必须使用相同的 `val_manifest.jsonl`、相同 tIoU threshold 和 temporal-only
event matching。S1a 的 `text` 字段只是事件类型占位符，因此不能将 text-gated caption 指标、
WER、歌词 CER 或 pointer accuracy 与 B3 比较。当前阶段有效的结论仅限：

- 是否更少遗漏重叠事件；
- 是否降低事件幻觉；
- activity mask 是否改善 SegF1 和边界误差；
- 性能是否随 source count/overlap 更缓慢退化。

只有完成 event-to-track pointer 和 slot-conditioned text decoder 后，才进入统一 caption 的公平
比较。S1a 结果不佳时，先检查 per-type recall、缓存身份、source leakage 和 slot eventness，不直接
据此否定完整 Hybrid Track--Event Ledger。

checkpoint 与 eventness threshold 只由 train 内的 calibration fold 决定，B3-valid validation
只用于本阶段 go/no-go。它仍不是论文最终测试集：方法冻结后必须另建 source-disjoint
test/WildMix-Cap，并且只运行一次最终评测；不能把这里的 validation 数字改名为 test result。

## 8. 服务器结果回传

服务器运行结束后，只需提交小型证据文件，不要提交模型和逐样本大型特征：

```bash
git add runs/s1_valid/model/run_manifest.json \
        runs/s1_valid/model/split.json \
        runs/s1_valid/model/run_summary.json
git commit -m "reports: add S1a-valid server result"
```

`runs/` 默认在 `.gitignore` 中。如果要提交正式报告，建议把上述三个文件复制到
`reports/s1a_valid/<run-id>/`，并同时记录服务器命令和模型权重校验值。不要提交
`best.pt`、`last.pt`、feature cache 或原始受限音频。
