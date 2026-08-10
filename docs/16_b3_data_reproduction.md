# B3-valid 数据复现与冻结协议

本文档只回答“训练输入是否已经被可靠复现”，不讨论模型分数。正式 B3/S1 训练必须读取
`data_reproduction_summary.json`，且其中 `pass=true`。旧 `data/derived/b3_unified` 的 500 条
synthetic vocal 数据不满足本协议。

## 1. 数据身份链

一次 B3-valid 发布由四个稳定哈希共同确定：

1. canonical source catalog；
2. renderer `manifest.jsonl`；
3. frozen `train_manifest.jsonl`；
4. frozen `val_manifest.jsonl`。

验收器将它们组合为 `dataset_id`。重新运行验证不会改变 `dataset_id`；任何音源、scene、target
或 split 改动都会产生新的 ID。checkpoint、prediction 和论文表格必须同时记录这个 ID。

## 2. 前置条件

- TAG 2021 锚点已经生成 `runs/tag2021/reproduction_summary.json`，且 `pass=true`；
- 五类 catalog 均已准备：`speech/vocal/music/sfx/ambience`；
- 每条 vocal 的 `text` 是音频实际包含的歌词，且 `verbatim=true`；
- 每条数据有明确 license/内部授权状态；
- 同一说话人、歌曲、原视频或衍生 stem 使用相同 `source_group`。

LibriSpeech 下载与登记见 `scripts/data/README.md`。需要申请或禁止自动分发的数据只提供登记流程，
脚本不会绕过许可下载。

## 3. 分阶段运行

首次先使用小规模 smoke，不直接渲染 10000 条：

```bash
SOURCE_CATALOG=/data/b3/source_catalog.csv \
SOURCE_AUDIO_ROOT=/data/b3/audio \
WORK_DIR=/data/runs/b3_smoke \
N_SAMPLES=100 \
bash scripts/run_b3_data.sh
```

完整流程依次执行 `sources → render → export → audit`。中断后可单独恢复：

```bash
STAGE=sources bash scripts/run_b3_data.sh
STAGE=render  bash scripts/run_b3_data.sh
STAGE=export  bash scripts/run_b3_data.sh
STAGE=audit   bash scripts/run_b3_data.sh
```

`sources` 阶段需要 `SOURCE_CATALOG` 和 `SOURCE_AUDIO_ROOT`；后续阶段只读取 `WORK_DIR` 中已经
冻结的产物。smoke 全部通过并完成试听后，再换新目录运行正式 10000 条：

```bash
SOURCE_CATALOG=/data/b3/source_catalog.csv \
SOURCE_AUDIO_ROOT=/data/b3/audio \
WORK_DIR=/data/runs/b3_valid_10k \
N_SAMPLES=10000 \
bash scripts/run_b3_data.sh
```

不要在同一个正式目录里用不同 `N_SAMPLES`、catalog 或 config 覆盖已有数据。

## 4. 产物与验收条件

| 产物 | 作用 |
|---|---|
| `source_catalog.jsonl` | 规范化的五类真实单源 catalog |
| `source_catalog_report.json` | 输入/输出 hash、kind、license、source group 与歌词统计 |
| `data/manifest.jsonl` | mixture、components、Ledger 与 renderer identity |
| `data/validation_report.json` | replay、stem、落盘 PCM、hash、Ledger 的逐批验收总结 |
| `sft/train_manifest.jsonl` | 冻结训练 scenes |
| `sft/val_manifest.jsonl` | 冻结验证 scenes |
| `sft/metadata.json` | split seed、manifest hash、泄漏与 placeholder 统计 |
| `data_reproduction_summary.json` | 所有检查和最终 `dataset_id` |

最终 summary 必须同时满足：

- 五类 source 数量均大于 0；
- 所有 vocal 都有 verbatim lyrics；
- 未允许缺文件，unknown license 数量为 0；
- rendered sample 数等于 `N_SAMPLES`；
- replay、stem sum、Ledger、落盘 reconstruction 全量通过；
- audio/hash/reconstruction failure 全部为 0；
- exporter 使用的 manifest hash 与 renderer 验证一致；
- missing audio、placeholder lyrics、source leakage 全部为 0；
- train/val sample IDs 和传递 source groups 均不相交；
- train+val 恰好覆盖全部 scenes；
- split manifest hash 与 `metadata.json` 一致。

验收器会重新读取两个 split 并重新计算 leakage，不信任 metadata 中自报的 0。若确有只能内部
使用且暂未补写 license 的 smoke 数据，可设置 `ALLOW_UNKNOWN_LICENSE=1`；这种数据不得作为论文
正式数据发布。

## 5. 人工检查

自动 gate 通过仍不等于语义标签正确。smoke 阶段至少按 template、事件类型、source count、
overlap ratio、T60 和 SNR 分层试听，记录：

- speech/lyrics 是否确实可听；
- `<lys>` 是否与演唱逐字一致；
- music、ambience 与 SFX 类别是否合理；
- onset/offset 是否覆盖可听区间；
- echo/reverb tail 是否被 residual 正确吸收；
- 多说话人和同歌 vocal/accompaniment 是否保持正确 source group。

当前 `listen_list.csv` 是抽查入口；正式 10k 冻结前应把人工结论另存为带 reviewer 和
`dataset_id` 的审计表。

## 6. 下游强制门槛

数据通过后才运行 B3：

```bash
STAGE=train WORK_DIR=/data/runs/b3_valid_10k \
B3_DATASET_ID=<accepted-dataset-id> \
MODEL_DIR=/models/MOSS-Audio-4B-Instruct \
bash scripts/run_b3_valid.sh

STAGE=infer WORK_DIR=/data/runs/b3_valid_10k \
B3_DATASET_ID=<accepted-dataset-id> \
MODEL_DIR=/models/MOSS-Audio-4B-Instruct \
bash scripts/run_b3_valid.sh

STAGE=evaluate WORK_DIR=/data/runs/b3_valid_10k \
B3_DATASET_ID=<accepted-dataset-id> \
bash scripts/run_b3_valid.sh
```

`run_b3_valid.sh` 和 `run_s1_valid.sh` 都会调用 `require_b3_data_pass.py`。summary 缺失、任何检查
失败或环境变量 `B3_DATASET_ID` 与 summary 不一致时，训练应停止。正式实验必须显式设置该变量，
把模型结果绑定到已验收的数据版本；未设置时仍会验证 summary 中存在非空 `dataset_id`。

## 7. 当前状态边界

代码和 CPU fixture 已覆盖 catalog、renderer、SFT split 与最终 acceptance gate；服务器尚未产生
正式 source catalog、100 条真实 smoke summary 或 10000 条正式 `dataset_id`。因此当前状态是
“复现代码就绪”，不是“数据复现完成”。最新 S1a-v3 在旧 synthetic B3 上得到的结果只作为
探索记录，不能用来决定架构去留。
