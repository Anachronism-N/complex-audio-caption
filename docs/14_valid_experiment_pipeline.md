# Anchor-first 与 B3-valid 实验执行手册

本文档是服务器实验的唯一执行入口。目标不是继续堆叠未经验证的尝试，而是依次建立
可复现锚点、可信训练集和只在验证集上报告的统一 caption 基线。

## 1. 实验顺序与停止条件

### R0/R1：TAG 2021 锚点

先执行 `repro/tag2021/README.md` 的 0--6 阶段。只有
`runs/tag2021/reproduction_summary.json` 中 `pass=true`，才进入 B1/B2/B3。
TAG 是 grounding 锚点，不是最终统一 caption 模型；它的作用是确认数据、特征、训练、
评价和随机种子链路都可由本项目稳定控制。

### B1：官方 MOSS SFT 锚点

重新渲染 renderer v0.3 数据后执行：

```bash
MODEL_DIR=/path/to/MOSS-Audio-4B-Instruct \
AUDIO_DIR=/path/to/tac_mini_v3 \
bash scripts/run_b1_official.sh
```

B1 使用 MOSS 官方 `finetune/finetune.py`，只验证静态 grammar 学习能力。验收条件：

- source-group 连通分量划分后 train/val 泄漏为 0；
- strict-format success 不低于 0.99；
- 所有指标仅来自 `val_manifest.jsonl`；
- inference parse report 必须传入 evaluator，不能把已解析 Ledger 默认算作 100% 格式成功。

### B2：0.1 秒原子时间戳与加权 CE

B2 使用项目自定义 trainer。301 个 `<|t_000|>`--`<|t_300|>` 是输出侧 token；它们不同于
MOSS 输入音频中每 2 秒插入的 time marker。训练器会：

1. 检查每个时间戳是否为单个 tokenizer ID；
2. 在配置允许时注册缺失 token 并 resize embedding；
3. 使用 PEFT `trainable_token_indices` 只训练新增的 301 行 embedding，并保持 tied lm-head；
4. 保存 `atomic_timestamp_tokens.json`；
5. 推理加载 LoRA 前按相同顺序恢复 vocabulary，并核对 token ID。

若上述任一步失败，训练应直接停止，不能退化为对标点和数字子 token 加权。

### B3-valid：统一 speech / lyrics / music / sfx

```bash
SOURCE_CATALOG=/path/to/source_catalog.csv \
SOURCE_AUDIO_ROOT=/path/to/source_audio \
MODEL_DIR=/path/to/MOSS-Audio-4B-Instruct \
N_SAMPLES=10000 \
MAX_STEPS=10000 \
bash scripts/run_b3_valid.sh
```

脚本顺序为：source catalog 审计、真实音源混合、确定性验证、无泄漏划分、MOSS SFT
导出、B3 训练、验证集推理、temporal-only 与 text-gated 两套评价。任何阶段失败都会停止。

## 2. Source catalog 契约

输入支持 CSV 或 JSONL，每行对应一个可独立混合的干声源：

| 字段 | 必需 | 含义 |
|---|---:|---|
| `path` | 是 | 音频文件，相对路径按 `--audio-root` 解析 |
| `kind` | 是 | `speech/vocal/music/sfx/ambience` |
| `text` | speech/vocal 必需 | 音频实际包含的转写、歌词或可验证描述 |
| `source_group` | 是 | 原说话人、歌曲或原媒体 ID；同组绝不跨 train/val |
| `identity` | 否 | 原始身份元数据；输出训练使用场景内 `S1/V1` slot |
| `language` | 否 | BCP-47/ISO 风格语言标记 |
| `verbatim` | vocal 必须为 true | 确认歌词逐字存在于音频中 |
| `license` | 建议必填 | 数据许可或内部研究授权状态 |
| `dataset` | 建议必填 | 来源数据集名称 |

先用以下命令独立检查：

```bash
python -m sceneledger.cli.prepare_sources \
  --input source_catalog.csv \
  --audio-root /data/source_audio \
  --output /data/source_catalog.jsonl \
  --report /data/source_catalog_report.json
```

`vocal` 若没有真实文本或 `verbatim=true` 会直接失败。听不清或无词人声应标为 `music`，
描述为 `unclear vocals/wordless vocals`，不能伪造 `<lys>`。

`configs/data/b3_real.yaml` 的完整模板需要 catalog 同时包含 `speech`、`vocal`、`music`、
`sfx` 与 `ambience`。不同语料先分别登记，再用重复的 `--input` 合并并执行一次总审计：

```bash
python -m sceneledger.cli.prepare_sources \
  --input /data/librispeech/source_catalog.jsonl \
  --input /data/opencpop/source_catalog.jsonl \
  --input /data/music_sfx/source_catalog.jsonl \
  --output /data/b3/source_catalog.jsonl \
  --report /data/b3/source_catalog_report.json \
  --require-kind speech --require-kind vocal --require-kind music \
  --require-kind sfx --require-kind ambience
```

推荐首轮公开数据组合：speech 使用 LibriSpeech/MLS 等带转写数据；lyrics 使用明确提供
音频与歌词许可的 singing corpus（例如按各自许可取得的 Opencpop、NUS-48E 或内部授权
数据）；music/sfx 使用许可兼容的 isolated stems。下载器只保存公开来源的 provenance，
不在仓库重新分发受限音频。

LibriSpeech 可直接下载、校验、解压并登记；默认下载 `train-clean-100`，首次冒烟测试建议用
较小的 `dev-clean`：

```bash
LIBRISPEECH_SUBSET=dev-clean \
LIBRISPEECH_ROOT=/data/librispeech \
bash scripts/data/download_librispeech.sh
```

已有数据可用 `sceneledger-catalog-librispeech` 直接生成 catalog。Opencpop 等需要申请的
singing corpus 不自动抓取；按官方许可取得后，用 `sceneledger-prepare-sources` 登记。
完整命令和许可边界见 `scripts/data/README.md`。

## 3. Source-aware atomic target

B3-valid 目标格式为：

```text
<speech track="T1" identity="S1"><|t_007|>hello<|t_029|></speech>
<speech track="T2" identity="S2"><|t_018|>good morning<|t_041|></speech>
<lys track="T3" identity="V1"><|t_032|>take me home<|t_061|></lys>
```

`track` 表示持续声源，`identity` 是场景内匿名 slot。旧格式仍能解析，但会按事件类型合并
track，不能用于多说话人实验。新的 formatter/parser round-trip 必须保持两个 speech tracks。

## 4. 数据划分

`group_split` 对共享 `source_group` 的场景构造连通分量。例如场景 A 使用歌曲 X 的 vocal，
场景 B 使用歌曲 X 的 accompaniment，即使文件路径不同，两者仍被放入同一 fold。导出器
同时写出：

- `train_manifest.jsonl`、`val_manifest.jsonl`；
- `train.jsonl`、`val.jsonl`；
- `val_references.jsonl`；
- `split.json`、`metadata.json`。

`metadata.json` 的 `source_leakage_count` 必须为 0。禁止对全部 manifest 推理后报告指标。

## 5. 评价与论文报告

每次实验至少保留：

- `val_infer_report.json`：raw output、clipping/rejection、格式成功率、退化输出计数；
- `val_metrics_temporal.json`：兼容历史结果的 type + tIoU 匹配；
- `val_metrics_text_gated.json`：额外要求 token-F1 不低于 0.1；
- per-type micro TP/FP/FN、边界 MAE/P90、hallucination、omission、source-count MAE；
- speaker-attributed WER/DER 和 lyrics CER/WER（后续 evaluator 接入前不得宣称完整解决）。

token-F1 gate 只是防止“文本完全无关却算 true positive”的最低保障，不是最终跨语言语义
指标。正式论文需冻结一个 multilingual embedding/LLM judge 版本，并用人工双盲子集报告与
人类判断的相关性。

### S1a-valid：事件槽定位探针

B3-valid 完成后再运行：

```bash
MODEL_DIR=/path/to/MOSS-Audio-4B-Instruct \
B3_WORK_DIR=/path/to/runs/b3_valid \
bash scripts/run_s1_valid.sh
```

该实验复用 B3-valid 的冻结 train/val manifests 和 MOSS features，训练 permutation-invariant
event slots，输出事件类型与多段 100 ms activity。实现、消融、输出文件和可声称范围见
`docs/15_s1_event_slot_experiment.md`。它不包含 caption text 或 track identity，不能把事件 F1
解释为统一 caption 质量。

## 6. 当前可声称与不可声称内容

代码就绪后可以声称：数据和目标格式能够保留四类事件、0.1 秒离散时间戳与多个 source
tracks，并且具有 fail-closed 的歌词监督和无泄漏划分。

在服务器实际运行完成前不能声称：模型达到 0.1 秒真实边界误差、复杂真实场景优于 TAC、
多说话人归因有效、歌词识别有效，或 hallucination 已被解决。旧 B3 报告已明确 supersede。
