# 数据管线复现状态与下一步实验

状态更新日期：2026-08-10。这里区分“代码已实现”“CPU 已验证”“GPU/真实数据已验证”，避免把存在脚本误写成已经完成实验。当前唯一服务器总入口是
`docs/14_valid_experiment_pipeline.md`；B3 数据冻结见 `docs/16_b3_data_reproduction.md`；S1a 见
`docs/15_s1_event_slot_experiment.md`。

## 1. 当前结论

TAC-style 合成数据的最小闭环已经实现：scene graph → source rendering → activity/span → Track–Event Ledger → mixture/stems/manifest → replay/audit → MOSS 官方 SFT JSONL。CPU 可以验证确定性、结构、分轨重构和无泄漏 split。

但是，仓库原先提交的 500 条 `data/derived/tac_mini/manifest.jsonl` 是旧 renderer 产物，存在两个 source-ID 碰撞和粗粒度 overlap 统计错误，也没有 residual stem。新审计器会拒绝它。正式 B1/B2 训练前必须用当前代码重新渲染，不能沿用旧 manifest。

真实单源语料、互联网无标注视频、teacher fusion、TAC++、CARC 和 WildMix-Cap 人工测试集尚未完成端到端服务器复现。

## 2. 分阶段状态

| 阶段 | 实现状态 | 验证状态 | 当前限制 |
|---|---|---|---|
| P0 schema/parser/serializer | 已实现 | CPU fixtures 已验证 | canonical 时间仍需从“静默量化”升级为严格校验 |
| P0 temporal/event metrics | 已实现 | CPU 已验证 | 已有 text hard gate；最终 multilingual semantic metric 待冻结 |
| P2 SyntheticSourcePool | 已实现 | 跨进程确定性已验证 | 只是占位波形，不能代表真实声学分布 |
| P2 scene templates | TAC-mini 与 B3-real 模板已实现 | CPU 已验证 | 真实数据分布仍需服务器统计与校准 |
| P2 RIR/echo/repeat | 已实现 | CPU 已验证 | noise、codec、ducking、occlusion 尚未进入 main renderer |
| P2 exact components | semantic stems + residual | CPU/落盘 PCM 审计已实现 | 需要在服务器重渲染 500 条正式数据 |
| source leakage split | union-find 传递分组 | CPU 反例已验证 | 真实数据还需加入 media ID、uploader、performer、audio fingerprint |
| B0 MOSS zero-shot | 已运行旧 500 条 | 有 500 条 raw output | 旧运行不是完整 deterministic protocol，需在修复数据上 greedy 重跑 |
| B1 MOSS static SFT | 官方格式导出和启动脚本已实现 | CPU 只验证数据导出 | GPU checkpoint、完整 val 指标待服务器运行 |
| B2 TAC-style weighted CE | 301 个原子时间 token、PEFT embedding rows 和 reload 校验已实现 | CPU tokenizer/loss tests 已验证 | GPU B2 数字待服务器运行 |
| B3-valid | 分阶段 source-readiness/render/export/audit、真实歌词 fail-closed、统一 target、dataset ID gate 已实现 | CPU 音频 probe/去重/配额与 renderer/export/acceptance fixtures 已验证 | 服务器尚无真实 `source_pool_id`；在它通过前禁止 render 和训练 |
| S1a-valid event slots | train/calibration/val 无泄漏划分、activity+boundary 双头、coverage-aware eval/消融已实现 | CPU 契约测试通过，Torch 模型测试待服务器 | 当前只做事件类型/时间，不含 track/text |
| 真实公开单源数据 | LibriSpeech 下载/catalog 脚本已实现；受限 singing 数据手工登记 | 未在目标服务器完整下载/审计 | 许可证、路径、checksum 和配额待确认 |
| B站/Instagram/TikTok 数据 | 原始无标注数据可用 | 未进入训练闭环 | 必须先做授权索引、去重、AV 切片和 teacher 置信度分层 |
| WildMix-Cap 200 条人工集 | 规范已设计 | 未标注 | 这是论文真实性结论的关键阻塞项 |

## 3. 当前服务器执行顺序

### 3.1 安装 CPU/音频依赖并验收

```bash
python -m pip install -e ".[dev,audio]"
ruff check .
pytest
```

### 3.2 重新渲染 TAC-mini

旧 manifest 不可继续使用。重新渲染到新的、空的输出目录：

```bash
python -m sceneledger.cli.render \
  --config configs/data/tac_mini.yaml \
  --output-dir /tmp/tac_mini_v2 \
  --validate
```

必须满足：

- 500/500 deterministic replay；
- 每个 scene source ID 唯一；
- semantic stems + `__residual__` 重构落盘 mixture；
- mixture/stem 文件 SHA-256 正确；
- 500/500 ledger schema valid；
- overlap ratio 在 0.1、0.5、1.0 秒 supervision 下通过测试；
- train/val raw-source intersection 为 0。

### 3.3 准备官方 MOSS B1 数据

```bash
python -m sceneledger.cli.prepare_moss_sft \
  --manifest /tmp/tac_mini_v2/manifest.jsonl \
  --audio-base /tmp/tac_mini_v2 \
  --output-dir /tmp/sceneledger_b1_sft \
  --target-mode atomic \
  --style brief \
  --seed 20260808
```

产物包括：

- `train.jsonl` / `val.jsonl`：MOSS 官方 conversation 格式；
- `train_manifest.jsonl` / `val_manifest.jsonl`；
- `val_references.jsonl`；
- `split.json`；
- `metadata.json`：manifest hash、prompt、split seed、样本数和审计结果。

### 3.4 运行 B1

先把官方 MOSS 仓库 checkout 到脚本固定的 commit，然后运行：

```bash
bash scripts/run_b1_official.sh
```

可通过环境变量指定路径：

```bash
MOSS_DIR=/path/to/MOSS-Audio \
MODEL_DIR=/path/to/MOSS-Audio-4B-Instruct \
AUDIO_DIR=/tmp/tac_mini_v2 \
OUTPUT_DIR=/path/to/outputs/b1_official \
bash scripts/run_b1_official.sh
```

脚本会执行官方 LoRA 训练、greedy validation inference 和 Ledger evaluation。B1 的主要科学问题只是“模型能否稳定学习统一 grammar”，不能把它作为 SceneLedger 方法创新。

### 3.5 冻结 B3-valid 数据

先只验收真实音源池，不渲染：

```bash
SOURCE_CATALOG=/data/b3/source_catalog.csv \
SOURCE_AUDIO_ROOT=/data/b3/audio \
SOURCE_PROFILE=smoke \
WORK_DIR=/data/runs/b3_smoke \
N_SAMPLES=100 \
STAGE=sources \
bash scripts/run_b3_data.sh
```

只有 `source_readiness_report.json` 的 `pass=true`、`source_pool_id` 已冻结且单源人工试听通过，
才运行 `STAGE=render`。之后按 `render → export → audit` 逐环验收；不要一次启动训练。
只有 `data_reproduction_summary.json` 的 `pass=true` 且 mixture 人工试听通过，才在新目录渲染 10k。
下游 B3/S1 runner 会拒绝没有通过该 gate 的数据。分阶段恢复、artifact contract 和失败检查见
`docs/16_b3_data_reproduction.md`；当前唯一应执行的步骤见 `docs/17_source_pool_readiness.md`。

## 4. B1 go/no-go gate

满足以下条件才进入 B2：

- strict-format-success ≥ 99%；
- val event-F1 ≥ 0.85；
- 各类型 F1 都有报告，不能只报 macro；
- onset/offset MAE、p90 和 0.1/0.25/0.5/1.0 秒 collar 全部保存；
- train/val 无 raw-source 泄漏；
- 完整保存 config、split IDs、manifest hash、MOSS commit、model/tokenizer revision、checkpoint hash 和原始 prediction；
- 至少重复 3 个 seed 或明确将单 seed 定义为工程 smoke，而不是论文结果。

如果格式成功但真实音频内容 F1 很低，下一步应优先替换 SyntheticSourcePool 为公开真实单源数据，而不是增加模型结构。如果 synthetic val 很高、真实 pilot 很低，则说明主要瓶颈是 sim-to-real gap，应进入 TAC++/真实 teacher/CARC，而不是继续在合成集刷分。

## 5. B2 当前状态

`<|t_000|>` 到 `<|t_300|>` 的工程门槛已经实现：

1. 注册 301 个 timestamp special tokens 和结构/type tokens；
2. resize input/output embeddings；
3. 断言每个时间 token 的编码长度严格等于 1；
4. 让新增 embedding rows 与 lm-head rows 可训练并正确保存；
5. 使用 shifted、权重归一化的 causal CE；
6. tokenizer/checkpoint reload 后 token ID 完全一致；
7. ordinary CE 与 weighted CE 在权重为 1 时数值一致。

上述条件由 tokenizer/loss 单元测试和训练入口 fail-closed 校验覆盖。仍未完成的是在固定 B3-valid
fold 上运行 GPU 训练、保存模型/tokenizer revision 与 checkpoint hash，并报告 validation 结果。
因此可以运行 B2，但在服务器产物回传前不能声称 B2 已复现成功。
