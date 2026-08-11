# Anchor-first：完整复现 TAG 2021 后再推进 SceneLedger

> 状态冻结于 2026-08-09。本文件是当前最高优先级实验协议；在 R0–R3
> 通过前，原有 B1/B2 只保留为工程探索结果，不作为论文证据。

## 1. 为什么把锚点固定到论文时代版本

复现对象是 ICASSP 2021 的 **Text-to-Audio Grounding: Building
Correspondence Between Captions and Sound Events**。任务输入一段音频和一个自然语言
声音短语，输出该声音对应的一个或多个起止区间。论文报告 Event-F1 28.3%、Precision
28.6%、Recall 27.9%、PSDS 14.7%。此外，作者把同一音频内的查询短语随机替换后仍得到
19.6% Event-F1，说明原模型明显依赖显著声学片段、没有充分使用文本条件。

这一诊断正好提供后续论文的第一个确定问题：**如何让时间定位真正受事件语义约束，而不是
在任何查询下都激活相同的显著片段**。

官方仓库后续切换到了 AudioGrounding v2、新的数据格式和新的训练实现。如果用 2025 年
代码、v2 数据却直接比较 2021 年论文数值，结果无法解释。因此本复现固定：

- 官方仓库 commit：`048f7af3d5167eeee0b7fd59aa877f46f245ff36`；
- 论文时代仓库内置 train/val/test 标签；
- 官方发布的 paper-era Google Drive 音频归档；
- 论文中的 64 维 log-mel、40 ms 窗、20 ms 帧移、CRNN + 平均词嵌入；
- 阈值 0.5、100 ms event collar、20% offset tolerance 和官方 PSDS 参数。

所有固定项同时写入 [`configs/reproduction_anchor.yaml`](../configs/reproduction_anchor.yaml)
和 [`repro/tag2021/upstream.lock.yaml`](../repro/tag2021/upstream.lock.yaml)。

## 2. 复现门槛

### R0：数据闭环

必须同时满足：

| split | 短语行数 | 唯一音频数 |
|---|---:|---:|
| train | 12,373 | 4,489 |
| val | 451 | 31 |
| test | 1,161 | 70 |

另外要求：

1. 所有标签中的音频都能按 basename 唯一解析；
2. train/val/test 没有音频交叉；
3. 所有时间段满足 `0 <= onset <= offset <= 10.1`；
4. 记录音频归档的文件 ID、字节数和 SHA-256；
5. `data_audit.json` 中 `valid=true`。

上游没有发布归档 checksum，所以不能假装存在官方哈希。我们的处理是第一次下载后生成
SHA-256，此后服务器间必须核对该值，出现不一致就不合并结果。

### R1：单种子核心结果

seed 1 必须满足：

| 指标 | 论文值 | 验收绝对误差 |
|---|---:|---:|
| Event-F1 | 0.283 | 0.03 |
| Precision | 0.286 | 0.03 |
| Recall | 0.279 | 0.03 |
| PSDS | 0.147 | 0.03 |

误差范围不是为了放宽模型标准，而是容纳旧 CUDA/cuDNN、PyTorch 和 GPU 架构带来的非完全
位一致性。如果超出范围，先定位数据、特征和依赖差异，不允许直接开始改模型。

### R2：论文诊断

对 test 集每一条短语，从同一音频的全部短语中有放回地随机采样查询，但保持原时间 GT
不变。论文没有公开随机种子，本实现固定 seed 1；采样可能得到原短语，因此报告中同时记录
实际改变的行数。期望 Event-F1 为 0.196，绝对误差不超过 0.04。包装代码会生成
`label_random_query_seed1.json`，并单独保存预测，避免覆盖正常测试结果。

### R3：多种子结果

运行 seeds 1/2/3，保存每个 checkpoint、原始预测、训练日志和指标；报告均值与标准差。
三个 seed 的均值需要满足 R1 门槛。`runs/tag2021/reproduction_summary.json` 的
`pass=true` 是允许开始创新实验的唯一机器可读开关。

## 3. 服务器执行

### 3.1 建环境

论文依赖非常旧，默认优先使用 legacy 环境。它最适合 V100/T4 等支持 CUDA 10.2 的机器：

```bash
conda env create -f repro/tag2021/environment-legacy.yml
conda activate tag2021-legacy
PYTHONPATH=src python -m sceneledger.repro.tag2021 doctor
```

脚本通过 `PYTHONPATH=src` 调用包装层，legacy 环境中不要执行 `pip install -e .`，因为主项目
其他模块以 Python 3.10 为最低版本。

如果服务器只有 A100/H100 或环境无法安装 PyTorch 1.6，使用兼容环境：

```bash
conda env create -f repro/tag2021/environment-modern.yml
conda activate tag2021-modern
```

使用 modern 环境得到的结果必须标为 `compatibility reproduction`。只有 legacy 结果可称为
依赖级复现；两种环境不能混合计算均值。若 modern 环境需要代码补丁，补丁必须单独提交，
并在报告中列出每一处修改。

`doctor` 会记录 Python、所有关键包版本、CUDA/cuDNN、GPU 型号，并把环境分类为
`legacy_exact` 或 `compatibility`；该环境快照也会自动写入每个 run 和评测结果。

### 3.2 下载与准备

```bash
bash scripts/repro/tag2021/00_bootstrap.sh
bash scripts/repro/tag2021/01_download.sh
bash scripts/repro/tag2021/02_prepare.sh
bash scripts/repro/tag2021/03_features.sh
```

如果 Google Drive 自动下载失败：

```bash
bash scripts/repro/tag2021/01_download.sh /path/to/AudioTextGrounding.zip
```

如果音频已经解压：

```bash
bash scripts/repro/tag2021/02_prepare.sh /path/to/extracted/root
```

准备器不会信任归档内标签，而是使用固定 commit 中的论文标签，并把 `filename` 解析为当前
服务器的绝对路径。它还会拒绝重复 basename、缺音频、非法时间段和 split 泄漏。

主要数据产物：

```text
external/tag2021/paper2021/
├── downloads/
│   └── audio_archive_provenance.json
├── raw/
└── prepared/
    ├── train/label.json
    ├── val/label.json
    ├── test/label.json
    ├── test/label_random_query_seed1.json
    ├── test/meta.csv
    ├── all_labels.json
    ├── data_audit.json
    ├── vocab.pkl
    └── logmel.hdf5
```

`external/` 被 gitignore；不得把受版权约束的原始音频推送到 GitHub。

### 3.3 训练与评测

```bash
for seed in 1 2 3; do
  bash scripts/repro/tag2021/04_train.sh "${seed}"
  bash scripts/repro/tag2021/05_evaluate.sh "${seed}"
done
bash scripts/repro/tag2021/06_summarize.sh 1 2 3
```

单 seed 训练用一张 GPU。建议预留 8 CPU、32 GB 内存、15 GB 本地磁盘；特征抽取主要消耗
CPU。首次执行不要同时提交三个训练任务：先让 seed 1 通过 R1，再提交 seeds 2/3。

Slurm 集群可在 R1 通过后使用：

```bash
sbatch repro/tag2021/slurm_train_array.sbatch
```

提交前先创建 Slurm 日志目录：`mkdir -p runs/tag2021/slurm`。

### 3.4 结果目录

```text
runs/tag2021/
├── configs/paper2021_seed1.yaml
├── seed_1/
│   ├── run.json
│   └── metrics.json
├── seed_2/
├── seed_3/
└── reproduction_summary.json
```

真正的 upstream checkpoint 和逐阈值预测位于 `run.json` 指向的 experiment 目录。正常查询和
随机查询分别保存在 `predictions_paper/` 与 `predictions_random_query/`。

## 4. 常见失败与判定

### 数据行数正确但音频数不正确

通常是归档没有完整解压，或用了 v2 音频命名。不要修改 expected count 绕过审计。确认下载
的是 Google Drive file ID `1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs`。

### Event-F1 接近论文、PSDS 明显偏低

优先检查：

1. 是否使用论文的 20 ms 帧移；
2. `test/meta.csv` 是否来自固定 commit；
3. `psds-eval` 版本是否为 legacy 的 0.3.0；
4. 是否误用了 v2 evaluator 或 v2 标签。

### modern 环境出现 API 错误

先保留完整 traceback。只允许做最小兼容补丁，例如 pandas/ignite API 迁移，不允许改变模型、
损失、特征或阈值。兼容补丁结果要与 legacy seed 1 交叉验证。

### 指标未过门槛

该结果就是复现失败，不得把容差继续放宽。按数据哈希 → 特征尺寸与采样率 → 配置 → 依赖 →
GPU 非确定性顺序排查，并在 `STATUS.yaml` 记录原因。

## 5. 复现完成后的第一个创新实验

R3 通过后只改变一个变量：保留 AudioGrounding v1 数据、query 输入、时间标签和官方 evaluator，
把旧 CRNN/平均词嵌入替换为现代音频-文本 grounding backbone。第一张论文表应为：

| 模型 | 正常 Event-F1 | 随机查询 Event-F1 | Query sensitivity gap | PSDS |
|---|---:|---:|---:|---:|
| TAG 2021 official | 复现值 | 复现值 | normal − random | 复现值 |
| modern backbone | 待测 | 待测 | 待测 | 待测 |

目标不是只提高正常 F1，而是在提高 F1/PSDS 的同时显著扩大 query sensitivity gap，证明模型
确实利用了查询语义。之后才按以下顺序扩展：

1. 单 query → 同一音频多 query；
2. 多 query → 一次输出所有 `<sfx>` 事件；
3. 加入 `<speech>/<music>/<lys>`；
4. 引入 SceneLedger 真实分布混音；
5. 最后研究反事实训练、幻觉奖励和 RL。

每一步都保留上一阶段数据和 evaluator，只改变一个主变量。这样即使新方法失败，也能明确
知道失败发生在哪一层。

## 6. 当前完成情况

本机没有实验算力，因此当前完成的是**可执行代码和无数据单元测试**，不是数值复现结果：

- upstream commit 锁定、下载与归档哈希记录：代码完成；
- label 解析、音频解析、split 泄漏与时间戳审计：代码完成并单元测试；
- log-mel、训练、官方 evaluator 和随机查询诊断包装：代码完成，等待服务器执行；
- R0/R1/R2/R3：均未宣称通过。

机器可读状态见 [`repro/tag2021/STATUS.yaml`](../repro/tag2021/STATUS.yaml)。
