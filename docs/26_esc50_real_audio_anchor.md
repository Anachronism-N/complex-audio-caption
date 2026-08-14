# 下一步实验：ESC-50 真实音频零样本锚点

更新日期：2026-08-13

## 1. 当前结论与本实验回答的问题

远端最新提交 `92a427e` 报告 MOSS-Audio 在随机 10 条 ESC-50 上人工判断约 6/10 正确。这是一个重要信号：此前 B3 数据失败的主要嫌疑确实是占位波形没有可识别语义，而不是 MOSS 完全不会理解真实音频。但该结果还不能作为论文实验，原因如下：

- 样本仅 10 条且不是按 50 类平衡抽样；“6/10”没有对应的原始结果文件和人工审核表。
- 数据路径、parquet 文件名和模型路径硬编码在 `/tmp`，换机器无法直接复现。
- 只测了自由文本 prompt，没有测试项目要求的 `<speech>/<lys>/<music>/<sfx>` 与 0.1 秒时间 token。
- ESC-50 每条约 5 秒、以单一环境声音为主；它不能证明复杂重叠、speech、music、lyrics 或精确边界能力。
- ESC-50 只有 clip-level 类别，没有事件起止真值；人工可以排除明显错误时间，但不能从该数据宣称 0.1 秒定位误差。

因此下一步不是立刻重新训练，也不是继续调 loss，而是完成一个 **50 类平衡、双 prompt、可审听、可验收的 MOSS 真实声源锚点**。它只回答两个前置问题：

1. 对真实 SFX/ambience，MOSS 的丰富语义 caption 是否达到可用于伪标注的最低水平？
2. 强制结构化输出后，语义、幻觉率和可用格式是否明显退化？

只有该门禁通过，才进入真实 source catalog 扩展和 100 条程序混音；未通过时应先修正教师模型或伪标注策略，不允许用失败数据训练 B3。

## 2. 新增实现

- `scripts/download_esc50.py`：从官方仓库固定 commit `33c8ce9...` 下载并安全解压。
- `sceneledger-prepare-esc50`：读取官方 `meta/esc50.csv`，也兼容已有 Hugging Face parquet；生成 strict source catalog，解码、哈希、查重并保留官方 fold。
- `sceneledger-caption-sources make-plan`：按 50 个类别确定性平衡抽样，为同一批音频冻结 semantic/structured 两种 prompt。
- `sceneledger-caption-sources run`：贪心推理，逐条落盘完整输出，支持中断后 `--resume`，并自动检查结构化输出能否解析。
- `sceneledger-caption-sources make-audit`：生成必须边听边填的人工审核 CSV。
- `sceneledger-caption-sources validate-audit`：校验完整性、重复行、推理错误、原始文本和计划哈希，输出机器可读的 go/no-go 报告。

ESC-50 的 `src_file` 被保存为 `source_group=freesound:*`，官方 fold 映射为 fold 1/2/3 → train、fold 4 → val、fold 5 → test。dataset label 仅作为弱标签写入 `labels` 与初始 caption，绝不伪装成人工丰富描述。

许可必须保留：完整 ESC-50 为 `CC BY-NC 3.0`，ESC-10 子集为 `CC BY 3.0`。这批数据可做非商业研究锚点，但不是未来可任意再分发的主训练语料。

## 3. 服务器逐步执行

以下假设仓库路径为 `$REPO`，实验输出为 `$RUN_ROOT`。先确认当前 Python 指向刚拉取的仓库，避免 editable install 串到另一个 checkout：

```bash
cd "$REPO"
git pull --ff-only
python -m pip install -e '.[data,dev]'
python -c 'import sceneledger, pathlib; print(pathlib.Path(sceneledger.__file__).resolve())'
python -m pytest tests/unit/test_esc50.py tests/unit/test_source_captioning.py tests/unit/test_source_catalog.py
```

最后一条 import 路径必须位于 `$REPO/src/sceneledger`。

### 3.1 数据准备（二选一）

方案 A：下载固定的官方版本。

```bash
export ESC_ROOT=/data/esc50
python scripts/download_esc50.py --output-dir "$ESC_ROOT"
export ESC_REPO="$ESC_ROOT/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
export ESC_META="$ESC_REPO/meta/esc50.csv"
export ESC_AUDIO="$ESC_REPO/audio"
```

方案 B：复用服务器已有的 Hugging Face parquet 与已解码 WAV。

```bash
export ESC_META=/tmp/real_audio/esc50/data
export ESC_AUDIO=/tmp/real_audio/esc50_wav
```

生成并验证全部 2,000 条 catalog：

```bash
export RUN_ROOT=/data/sceneledger_runs/esc50_anchor_v1
mkdir -p "$RUN_ROOT"
sceneledger-prepare-esc50 \
  --metadata "$ESC_META" \
  --audio-root "$ESC_AUDIO" \
  --output-dir "$RUN_ROOT/catalog"
python - <<'PY'
import json, os
p = os.path.join(os.environ['RUN_ROOT'], 'catalog/prepared/source_catalog_report.json')
r = json.load(open(p))
print('pass=', r['pass'])
print('counts_by_kind=', r['counts_by_kind'])
print('counts_by_split=', r['counts_by_split'])
print('failed=', [c for c in r['checks'] if not c['pass']])
assert r['pass']
PY
```

预期是 2,000 条、50 个 label、train/val/test 分别 1,200/400/400 条；所有 license、解码、音频哈希、跨 split 泄漏检查必须通过。

### 3.2 冻结 50 类 × 1 条 × 2 prompt 的计划

```bash
sceneledger-caption-sources make-plan \
  --catalog "$RUN_ROOT/catalog/prepared/all.jsonl" \
  --audio-root "$ESC_AUDIO" \
  --per-label 1 \
  --seed 20260813 \
  --output "$RUN_ROOT/caption_plan.json"
```

预期输出 `n_labels=50, n_sources=50, n_generations=100`。计划生成后不要编辑；它冻结样本、音频 SHA256、prompt 和实验条件。先用 50 类全覆盖代替继续报告随机 10 条。

### 3.3 MOSS 贪心推理

```bash
export MOSS_WEIGHTS=/tmp/moss_weights
sceneledger-caption-sources run \
  --plan "$RUN_ROOT/caption_plan.json" \
  --model-path "$MOSS_WEIGHTS" \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 1024 \
  --output "$RUN_ROOT/caption_results.jsonl" \
  --report "$RUN_ROOT/caption_run_report.json"
```

任务中断或个别推理报错后，原命令增加 `--resume`，不要删除已完成结果。CLI 会保留成功行并自动重试错误行。`run_report.pass` 必须为 true，`n_results` 必须为 100；失败项先处理，不能从审核表中删掉。

### 3.4 你必须做的人工工作

```bash
sceneledger-caption-sources make-audit \
  --results "$RUN_ROOT/caption_results.jsonl" \
  --output "$RUN_ROOT/caption_audit.csv"
```

你或另一名听力正常的审核者需要戴耳机逐条播放 `audio_path`。同一个音频有 semantic/structured 两行，可以连续审核；不能只看 ESC 类别和模型文本作判断。填写：

- `label_correct_y_n`：模型描述的核心声类是否正确。
- `all_audible_events_covered_y_n`：可辨认事件是否全部覆盖。
- `hallucination_free_y_n`：是否没有不可听见的物体、视觉场景、说话内容或声学属性。
- `temporal_claims_present_y_n`：输出是否包含时间、顺序、次数或持续性主张。
- `temporal_claims_supported_y_n_or_na`：有上述主张则填 `y/n`，没有则填 `na`。这里只判断听觉上是否明显成立，不把它当 0.1 秒真值。
- `structured_format_usable_y_n_or_na`：structured 行填 `y/n`，semantic 行固定 `na`。
- `corrected_caption` 和 `notes`：可选，但错误类型必须写清楚，便于决定下一步修复。

建议先让一位审核者完成 100 行，耗时约 1–2 小时；随后第二位审核者独立复核所有错误行和随机 20% 的通过行。当前 CLI 门禁绑定第一份完整表；双人一致性可在进入论文 benchmark 标注时另行统计 Cohen's kappa。

### 3.5 自动验收

```bash
sceneledger-caption-sources validate-audit \
  --plan "$RUN_ROOT/caption_plan.json" \
  --results "$RUN_ROOT/caption_results.jsonl" \
  --audit "$RUN_ROOT/caption_audit.csv" \
  --output "$RUN_ROOT/caption_audit_report.json"
```

默认 go 条件对两个 prompt 分别要求：类别正确率 ≥ 70%、事件覆盖率 ≥ 70%、无幻觉率 ≥ 80%；structured 还要求人工格式可用率和自动 parser 成功率均 ≥ 50%、至少 50% 输出作出时序主张，且这些主张中 ≥ 70% 获人工支持。这些阈值是进入伪标注 pilot 的工程门槛，不是论文 SOTA 标准。

## 4. 结果分支：不要无条件继续训练

### A. semantic 与 structured 都通过

说明 MOSS 可作为真实 SFX/ambience 的候选教师。下一步代码应扩展到许可合适的 speech、vocal、music 数据，并对每类先做同样 source-level 审核；五类都过 gate 后才构造 100 条真实 source 程序混音，并人工审听混音是否与 ledger 一致。

### B. semantic 通过、structured 不通过

说明音频理解能力可用，但零样本格式遵循/时间输出不可用。不要丢弃 MOSS；改走两阶段教师：MOSS 生成 evidence-rich 自由文本，已有 ESC label/事件检测器提供类别约束，时间边界由专门的音频 grounding/SED 模型或人工小样本产生，再做受约束重写。随后用少量通过审核的结构化样本进行 SFT，而不是直接训练 B3 全量数据。

### C. 两者均不通过

停止用 MOSS 自动扩写 source caption。优先比较更强 teacher 或加入已知 class label 的条件提示；也可以只保留官方 label 做事件检测数据，但不能把错误自由 caption 当训练真值。此时继续调模型 loss、RL 或复杂混音都没有因果解释力。

## 5. 本实验不能证明什么

即使通过，也只能证明“真实孤立 SFX/ambience 的零样本语义和初步格式能力”。它不能证明：多说话人分离、歌词转写、详细音乐风格、混响/噪声鲁棒性、多源重叠、track/event 数量预测，或 0.1 秒边界精度。后续实验必须按 source → 100 real-mix → complex benchmark 的顺序逐环验收，不能跨级归因。
