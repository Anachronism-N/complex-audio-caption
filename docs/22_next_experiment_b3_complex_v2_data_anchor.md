# 下一步实验：B3-complex-v2 数据锚点生成与验收

## 0. 本轮决策

下一步**不训练模型，也不新增模型代码**。先完整运行并验收 `B3-complex-v2` 数据锚点，确认以下基础事实：

1. renderer 能从固定配置稳定生成 train/val/test 三折；
2. 每条 mixture 都能从相应 stems 重放，Ledger 与音频一致；
3. 三折的 scene ID 和底层 source identity 完全隔离；
4. “十多秒只有一个短音效、后面长时间静音”不再成为主要分布；
5. 生成物、阈值、引用答案和哈希均被冻结，后续训练不能偷偷换数据。

现有代码已经覆盖这些要求：

- `scripts/run_b3_complex_v2_data.sh`：三折生成与总门禁；
- `configs/data/b3_complex_v2_{train,val,test}.yaml`：互斥 source 范围、固定 seed 和场景模板；
- `configs/data/mixture_quality.yaml`：`release` 质量阈值；
- `sceneledger.cli.render --validate`：deterministic replay、stems-sum、Ledger schema；
- `sceneledger.cli.validate_experiment_data`：split contract、source leakage、分布质量、reference 冻结。
- `sceneledger.cli.preflight_data`：写 WAV 前冻结完整 scene plan，检查源数分层和复杂模板覆盖。

因此本轮继续增加 renderer 或训练侧功能只会引入新变量。先取得一个明确的 PASS/FAIL 结论，再决定是否进入真实单源语料接入。

> 重要边界：当前配置使用 `SyntheticSourcePool`。其中 speech、vocal、music、sfx 是程序生成的占位波形，不是真实语音、歌曲或音效。这个实验验证的是**数据工程、时序布局和评测隔离协议**，不是训练语义 caption 模型所需的最终数据，也不能证明 TAC 已复现。

### 0.1 代码完备性与已完成验证（2026-08-12）

本轮审计结论是：**D0 不存在阻塞性代码缺口，不新增代码。** 新代码只有在正式 5000 条运行暴露出可复现的 renderer、contract 或质量统计错误时才允许加入；不能在尚未运行 D0 前继续修改数据分布。

已在当前工作区完成以下验证：

- `train/val/test` 各渲染 10 条，三折均为 replay `10/10`、stems-sum `10/10`、Ledger `10/10`、`failures=0`；
- 30 条烟测产物能够进入 `validate_experiment_data`，split contract 为 `pass=true` 且无 scene/source 交集；
- 对这 30 条运行分布门禁会因样本太少、某些 fold 没抽到 `repeated_event` 或 `overlapping_speakers` 而返回非零退出码。这是 fail-closed 门禁的预期行为，不是 D0 失败，也不能通过降低阈值处理；正式分布结论只能来自 4000/500/500 全量数据；
- 数据锚点关键文件从提交 `74c5566e496f8d072ed2068a05b92e6b1cd52d48` 到本次审计没有变化；后续 DPO 提交只增加了训练脚本、模型配置和报告，不影响本协议的 renderer 与门禁代码。

最近一次旧 `b3_5k` DPO 实验不能改变执行顺序：其 384 个偏好对全部与 500 条评测样本重叠，reference 不是冻结 SFT 模型，rejected 又丢失了原始格式错误；最终实际严格格式率只有 `49/500=9.8%`，Event-F1 约为 `0.0045`。因此在 D0 与后续真实单源数据锚点完成前，禁止继续 DPO、RL、CARC 或 MOSS 主训练。

## 1. 实验问题与可证伪假设

本实验只回答一个问题：**当前数据管线是否已经可靠到可以接入真实单源语料？**

| 假设 | 验证方式 | 通过条件 |
|---|---|---|
| H1：渲染可重放 | 每折执行 `render --validate` | replay、stems-sum、Ledger 全部通过，0 failure |
| H2：三折无泄漏 | 构建 split contract | scene ID 与 source identity 的三组交集均为空 |
| H3：复杂度不再退化 | `release` quality profile | 所有 fold 的 `failed_checks=[]` |
| H4：产物可冻结 | manifest/config/report/reference SHA-256 | summary 能通过二次完整性验证 |
| H5：混合逻辑听感正确 | 固定抽样人工试听 | 没有系统性长静音、错误重叠、不可闻事件或削波 |

H1–H4 任一失败都必须停止，不能提交 GPU 训练。H5 若出现系统性问题，也必须先定位模板或 renderer；不能通过降低门禁阈值掩盖问题。

## 2. 固定代码版本

`B3-complex-v2` 的远端实现位于分支 `agent/valid-data-protocol-v2`。本次运行至少应包含提交：

```text
74c5566e496f8d072ed2068a05b92e6b1cd52d48
```

在服务器的仓库目录执行：

```bash
git fetch origin agent/valid-data-protocol-v2
git switch --detach 74c5566e496f8d072ed2068a05b92e6b1cd52d48
git status --short
git rev-parse HEAD
```

预期：`git status --short` 没有输出，`git rev-parse HEAD` 与上面的 SHA 完全相同。实验期间不要切换分支、`git pull` 或修改配置。

如果以后用更新提交运行，应把新的完整 commit SHA 记录到实验目录，并为结果分配新的数据版本；不要继续沿用本实验的 `dataset_id`。

## 3. 服务器要求与环境安装

本阶段只需要 CPU，不需要 CUDA、GPU 或 MOSS-Audio 权重。建议使用 Linux、Python 3.10 或 3.11，并在 CephFS/本地高速盘预留至少 50 GB。输出包含 5000 条 mixture 及每条场景的多个 PCM-16 stem，小文件数量也会较多。

```bash
conda create -n sceneledger-data python=3.10 -y
conda activate sceneledger-data

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" soundfile
```

`soundfile` 是渲染 WAV 的实际运行依赖；基础 `pip install -e .` 当前不会自动安装它。若安装时报 `libsndfile` 错误，先在系统中安装 `libsndfile1`，再重试。此实验不需要安装 `torch`、`torchaudio` 或完整的 `.[moss]`。

安装后检查：

```bash
python - <<'PY'
import numpy, pydantic, scipy, soundfile, yaml
print("imports: PASS")
print("soundfile:", soundfile.__version__)
PY

df -h .
```

只有依赖导入成功且目标盘空间充足时才继续。

## 4. 运行前测试

在仓库根目录执行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash -n scripts/run_b3_complex_v2_data.sh
pytest -q tests/unit tests/integration/test_renderer.py
```

通过条件：shell 语法检查退出码为 0，pytest 没有失败。若测试失败，保留完整日志并停止；不要先生成 5000 条数据。

### 4.1 小规模渲染烟测

烟测只检查三份配置是否可执行，不用于估计正式分布，也不能代替 `release` 门禁：

```bash
export SMOKE_ROOT=/cephfs/your_project/sceneledger/b3_complex_v2_smoke_74c5566
mkdir -p "$SMOKE_ROOT"

for split in train val test; do
  python -m sceneledger.cli.render \
    --config "configs/data/b3_complex_v2_${split}.yaml" \
    --output-dir "$SMOKE_ROOT/$split" \
    --limit 10 \
    --validate
done
```

每折应看到类似：

```text
[validate] replay ok=10/10 stems_sum ok=10 ledger_valid=10 failures=0
```

不要对 10 条烟测运行 `release` 分布门禁，也不要因为存在名为 `smoke` 的 profile 就把 10 条结果当成分布验收。小样本可能完全缺失低权重模板，模板比例波动也没有统计意义。烟测目录必须与正式输出目录分开。

## 5. 正式生成与自动验收

正式脚本首先执行无 WAV I/O 的 scene-plan preflight，并写出
`$OUTPUT_ROOT/scene_plan_preflight.json`。三折必须同时满足：平均源数至少
3.4；simple/medium/complex（按实际 source count 定义）比例落在
15–30% / 40–60% / 20–40%；`complex_cocktail`、`rich_band`、
`multi_event_dense` 每个至少占 8%。任一项失败，脚本会在耗时渲染前退出。

preflight 保存每一折有序 scene dictionary 的 SHA-256。完成渲染后，
`validate_experiment_data` 会从 manifest 重算该哈希，禁止预检一套配置却渲染另一套配置。

选择一个新的、版本化且为空的输出目录。下面的路径只是示例，必须替换成服务器上的真实 CephFS 路径：

```bash
export OUTPUT_ROOT=/cephfs/your_project/sceneledger/b3_complex_v2_74c5566
if [ -e "$OUTPUT_ROOT" ]; then
  echo "Refusing to reuse existing output root: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

git rev-parse HEAD > "$OUTPUT_ROOT/git_commit.txt"
python --version > "$OUTPUT_ROOT/python_version.txt" 2>&1
python -m pip freeze > "$OUTPUT_ROOT/pip_freeze.txt"

set -o pipefail
PYTHON_BIN="$(command -v python)" \
  bash scripts/run_b3_complex_v2_data.sh "$OUTPUT_ROOT" \
  2>&1 | tee "$OUTPUT_ROOT/run.log"
run_status=${PIPESTATUS[0]}
echo "$run_status" | tee "$OUTPUT_ROOT/exit_code.txt"
test "$run_status" -eq 0
```

脚本会严格按以下顺序执行：

1. train：4000 条，synthetic source index `[0, 799]`；
2. val：500 条，synthetic source index `[800, 899]`；
3. test：500 条，synthetic source index `[900, 999]`；
4. 对每折进行音频重放、stems-sum 和 Ledger 验证；
5. 对三折执行 `release` 质量门禁并冻结 contract/reference。

脚本采用 fail-closed 行为：任一步退出码非 0，后续步骤停止。不要在外层添加 `|| true`，也不要因为接近 12 小时时限而跳过验证。

当前脚本没有断点续跑语义。如果运行中断，应先保存 `run.log` 和失败目录用于诊断；修复后使用新的版本化输出目录完整重跑，避免把部分旧文件误当作新结果。

### 5.1 12 小时执行窗口

建议按下面的时间预算执行。时间是监控预算而不是结果门槛；如果提前完成就立即进入验收，不需要等待满 12 小时。

| 时间 | 动作 | 必须保存的证据 | 停止条件 |
|---|---|---|---|
| 0:00–0:30 | 固定 commit、安装/核对环境、检查磁盘 | `git_commit.txt`、Python、`pip freeze` | 依赖失败、工作树不干净、空间不足 50 GB |
| 0:30–1:00 | 单元测试和三折各 10 条烟测 | 测试日志、三折 validate 摘要 | 任一 replay/stems-sum/Ledger 失败 |
| 1:00–10:00 | 正式 4000/500/500 生成与逐条验证 | `run.log`、三折 manifest/audio/stems | 非零退出、磁盘不足、进度长时间不增长 |
| 10:00–11:00 | split contract、质量门禁、hash/reference 冻结 | 完整 `gate/`、`exit_code.txt` | 任一 `pass=false` 或完整性检查失败 |
| 11:00–12:00 | 固定 60–80 条人工试听和结果打包 | audit CSV、data cards、压缩包 | 模板级系统错误或严重错误超过预注册上限 |

如果全量渲染超过预计时间，不要删除 stems、跳过 test 或跳过门禁来凑进 12 小时。保留日志和已有输出用于性能诊断，但该目录仍标记为 `INCOMPLETE`，不能用于训练；下一次在新的输出目录完整重跑。

## 6. 运行中监控

另开终端，只执行只读检查：

```bash
tail -f "$OUTPUT_ROOT/run.log"
```

```bash
du -sh "$OUTPUT_ROOT"
find "$OUTPUT_ROOT/train/audio" -maxdepth 1 -name '*.wav' | wc -l
find "$OUTPUT_ROOT/val/audio"   -maxdepth 1 -name '*.wav' | wc -l
find "$OUTPUT_ROOT/test/audio"  -maxdepth 1 -name '*.wav' | wc -l
```

renderer 每 50 条打印一次进度。正式完成后，三个 mixture 数量应分别为 4000、500、500；`audio/stems/` 下还有额外的源级文件。

若磁盘空间快速耗尽，应停止进程并换到更大的输出盘；不要删除正在运行目录中的 stems，因为 stems-sum 验证和后续可审计性依赖它们。

## 7. 自动结果判读

正式完成后，输出结构至少应包含：

```text
$OUTPUT_ROOT/
├── scene_plan_preflight.json
├── train/{manifest.jsonl,data_card.md,listen_list.csv,audio/}
├── val/{manifest.jsonl,data_card.md,listen_list.csv,audio/}
├── test/{manifest.jsonl,data_card.md,listen_list.csv,audio/}
├── gate/
│   ├── split_contract.json
│   ├── train_mixture_quality.json
│   ├── val_mixture_quality.json
│   ├── test_mixture_quality.json
│   ├── train_references.jsonl
│   ├── val_references.jsonl
│   ├── test_references.jsonl
│   ├── experiment_data_summary.json
│   ├── human_audit_tasks.csv
│   └── human_audit_tasks.meta.json
├── git_commit.txt
├── pip_freeze.txt
├── python_version.txt
├── run.log
└── exit_code.txt
```

先核对行数：

```bash
wc -l \
  "$OUTPUT_ROOT/train/manifest.jsonl" \
  "$OUTPUT_ROOT/val/manifest.jsonl" \
  "$OUTPUT_ROOT/test/manifest.jsonl" \
  "$OUTPUT_ROOT/gate/train_references.jsonl" \
  "$OUTPUT_ROOT/gate/val_references.jsonl" \
  "$OUTPUT_ROOT/gate/test_references.jsonl"
```

预期 manifest 与 reference 分别为 `4000/500/500`，且同一 fold 的两者行数相同。

再打印总门禁、split 检查与每折指标：

```bash
python - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
gate = root / "gate"
summary = json.loads((gate / "experiment_data_summary.json").read_text())
contract = json.loads((gate / "split_contract.json").read_text())

print("summary.pass:", summary["pass"])
print("dataset_id:", summary["dataset_id"])
print("failed_checks:", summary["failed_checks"])
print("split_contract.pass:", contract["pass"])
print("split_failed_checks:", contract["failed_checks"])
for split in ("train", "val", "test"):
    report = json.loads((gate / f"{split}_mixture_quality.json").read_text())
    print("\n", split, "pass=", report["pass"], "failed=", report["failed_checks"])
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print("templates:", {k: v["n"] for k, v in report["by_template"].items()})
PY
```

最后调用仓库中的严格加载器再次检查文件没有在门禁生成后被修改：

```bash
python - "$OUTPUT_ROOT" <<'PY'
import pathlib
import sys
from sceneledger.data.experiment_data import require_experiment_data_summary

root = pathlib.Path(sys.argv[1])
gate = root / "gate"
payload = require_experiment_data_summary(
    gate / "experiment_data_summary.json",
    gate / "split_contract.json",
)
print("integrity: PASS")
print("dataset_id:", payload["dataset_id"])
PY
```

### 7.1 `release` 阈值的准确含义

| 检查项 | 正式阈值 |
|---|---:|
| 单事件场景比例 | `<= 5%` |
| active ratio `< 0.30` 的比例 | `<= 10%` |
| 尾部静音 `> 5 s` 的比例 | `<= 10%` |
| 任意连续静音 `> 5 s` 的比例 | `<= 10%` |
| 场景内重复 source ID 比例 | `0%` |
| 场景内重复底层 source path 比例 | `0%` |
| 平均 source count | `>= 3.4` |
| simple / medium / complex 比例 | `15–30% / 40–60% / 20–40%` |
| 三个新增复杂模板各自比例 | `>= 8%` |
| complex 样本 overlap ratio `< 0.15` 的比例 | `<= 20%` |
| `isolated_sfx` 模板比例 | `<= 5%` |
| `repeated_event` 中少于 2 个 SFX span 的违规比例 | `0%` |
| `overlapping_speakers` 中 overlap ratio `< 0.10` 的违规比例 | `<= 10%` |

这些是当前工程验收阈值，不是 TAC 论文报告值，也不是对真实互联网音频分布的统计结论。若阈值设计不合理，应在分析失败原因后新建版本化 profile，不能直接修改 `release` 后覆盖旧结果。

## 8. 人工试听协议

**需要人工试听。** 自动门禁能够验证文件、时间调度、分布和 stems 数值关系，但无法证明事件实际可听、时间听感正确、重叠真实可辨，或者 mixture 没有削波与异常静音。由于本轮是 synthetic placeholder，试听不评价自然度、ASR 文本真实性或音乐风格细节；只评价以下机械属性：

- manifest 声称出现的源在相应时间是否可听；
- speech/music/vocal/sfx/ambience 的粗类别是否可区分；
- 重叠模板是否真的存在可听重叠；
- repeated event 是否能听到多次实例，而非仅 Ledger 重复；
- 是否有超长无意义静音、突兀截断、严重削波或近乎不可闻的前景；
- mixture 是否与各 stem 的听感组合一致。

正式脚本会在自动门禁通过后生成冻结的 `gate/human_audit_tasks.csv` 和对应 metadata。任务采用由 `dataset_id` 决定的稳定抽样，而不是“列表前五条”：

- test 集每个实际存在的模板抽 5 条；
- 额外纳入质量报告中最多 20 条风险样本；
- 同一样本只出现一次，但会同时记录 `template_stratified` 和 `quality_violation` 原因；
- 每个模板至少有一条要求同时试听各 stems 和 mixture；
- 含 track overlap 的样本必须评价重叠听感；
- `audit_id` 和 `tasks_sha256` 冻结样本、顺序及预期事件，不能手工换掉难例。

生成命令已经包含在 `scripts/run_b3_complex_v2_data.sh` 中，也可以在门禁通过后单独执行：

```bash
python -m sceneledger.cli.human_audit prepare \
  --manifest "$OUTPUT_ROOT/test/manifest.jsonl" \
  --data-gate-summary "$OUTPUT_ROOT/gate/experiment_data_summary.json" \
  --split-contract "$OUTPUT_ROOT/gate/split_contract.json" \
  --expected-split test \
  --per-template 5 \
  --max-violation-samples 20 \
  --output-csv "$OUTPUT_ROOT/gate/human_audit_tasks.csv" \
  --output-metadata "$OUTPUT_ROOT/gate/human_audit_tasks.meta.json"
```

用 Excel、LibreOffice 或支持 UTF-8 CSV 的工具填写以下 reviewer 字段；不可修改前面的任务字段：

| 字段 | 允许值 | 含义 |
|---|---|---|
| `reviewer` | 非空匿名 ID | 审听者，不写真实姓名也可以 |
| `reviewed_at_utc` | UTC 时间文本 | 审听完成时间 |
| `event_audibility` | `pass/fail/uncertain` | Ledger 中声明的事件是否都可听 |
| `timestamp_alignment` | `pass/fail/uncertain` | 声音出现区间是否与 0.1 s Ledger 基本一致；本阶段按听感判断明显错误，不要求人耳证明 0.1 s 精度 |
| `overlap_rendering` | `pass/fail/uncertain/not_required` | 有重叠任务必须填写前三者，无重叠必须填 `not_required` |
| `long_silence` | `absent/present/uncertain` | 是否存在不符合 Ledger/模板的长静音 |
| `clipping` | `absent/present/uncertain` | 是否存在明显削波或爆音 |
| `stem_mixture_consistency` | `pass/fail/uncertain/not_required` | 标为需要 stem review 时比较 stems 与 mixture，否则填 `not_required` |
| `severity` | `none/minor/severe` | 问题严重程度 |
| `overall_decision` | `pass/fail/uncertain` | 该样本总体结论 |
| `notes` | 自由文本 | 问题时间点和现象；失败样本应填写 |

填写完成后汇总：

```bash
python -m sceneledger.cli.human_audit summarize \
  --review-csv "$OUTPUT_ROOT/gate/human_audit_tasks.csv" \
  --metadata "$OUTPUT_ROOT/gate/human_audit_tasks.meta.json" \
  --max-severe 2 \
  --max-total-failures 2 \
  --template-failure-threshold 2 \
  --output "$OUTPUT_ROOT/gate/human_audit_summary.json"
```

该命令采用 fail-closed 判定，以下情况均返回非零退出码：

- 任一任务未完成或仍为 `uncertain`；
- 任务 ID、样本、路径、期望事件或顺序被修改；
- 必做的 overlap/stem 检查被填为 `not_required`；
- 单项失败却把总体结论填成 `pass` 等自相矛盾情况；
- severe 或总失败数超过预注册上限；
- 同一模板同一指标至少 2 条失败，视为模板级系统问题。

如果第一名审听者选择 `uncertain`，由第二名审听者复听并在 `notes` 中记录双方判断，最后把字段更新为裁决后的 `pass/fail`。不要直接把 `uncertain` 改成 `pass` 来通过门禁。

建议判定：

- **PASS**：全部任务完成、没有模板级系统性错误、严重错误和总体失败均不超过 2 条；
- **FAIL**：同一模板连续出现同类错误、Ledger 事件实际不可听、重叠/重复模板名不副实，或长静音问题仍普遍存在。

人工标准是本轮预注册的工程规则。若听感结果模棱两可，增加第二名审听者并记录分歧，不要让单人主观印象直接决定是否训练。

## 9. 最终决策表

| 条件 | 决策 |
|---|---|
| H1–H5 全部通过 | 数据工程锚点成立；下一步才接入真实单源语料 |
| replay 或 stems-sum 失败 | 修 renderer/量化容差，不训练 |
| split/source disjoint 失败 | 修 source-group 划分，不训练 |
| 复杂度门禁失败 | 定位失败模板与采样逻辑，不降低阈值凑 PASS |
| 自动门禁通过但人工试听失败 | 自动指标覆盖不足；补充可证伪检查后重跑 |
| 仅 synthetic 数据通过 | 只能声称 pipeline validation，不能声称数据集或模型复现成功 |

通过后应冻结整个 `gate/`、三份 manifest、三份配置、`mixture_quality.yaml`、完整 commit SHA 和 `dataset_id`。后续任何训练命令都必须通过 `--experiment-data-summary` 与 `--split-contract` 引用这些冻结文件。

## 10. 需要回传的最小结果包

无需上传全部 WAV。请回传以下内容即可完成远程分析：

1. `git_commit.txt`、`python_version.txt`、`pip_freeze.txt`；
2. `run.log`、`exit_code.txt` 与 `scene_plan_preflight.json`；
3. 完整 `gate/` 目录；
4. train/val/test 的 `data_card.md` 和 `listen_list.csv`；
5. 已填写的 `human_audit_tasks.csv` 与 `human_audit_summary.json`；
6. 若失败，再附失败 scene 的 manifest 行及对应 mixture/stems。

建议压缩命令：

```bash
tar -czf b3_complex_v2_74c5566_reports.tar.gz \
  -C "$OUTPUT_ROOT" \
  git_commit.txt python_version.txt pip_freeze.txt run.log exit_code.txt \
  gate \
  train/data_card.md train/listen_list.csv \
  val/data_card.md val/listen_list.csv \
  test/data_card.md test/listen_list.csv
```

## 11. 通过后的唯一下一步

本锚点通过后，下一阶段不是立即用这 5000 条 placeholder 音频训练 MOSS，而是：

1. 确定可合法使用的真实单源 speech、music、vocal/lyrics、SFX、ambience 语料；
2. 为每条真实 recording 建立稳定的 `source_group`、许可信息、文本/类别标签和原始切分；
3. 按 recording/speaker/song/uploader 等 group 先分 train/val/test，再混合，严禁先混合后随机切分；
4. 用同一 renderer、split contract 和质量门禁生成 `B3-real-v1`；
5. 对真实混合数据完成试听和小规模过拟合测试后，才提交正式 GPU 训练。

该阶段是否需要新代码，取决于真实语料目录和元数据格式。当前不能在不知道语料 schema 的情况下盲写 adapter；收到各语料的 manifest 示例后，再实现 `FileSourcePool` 数据适配、group split 和 caption/时间标签映射。

## 12. 实验结果记录模板

运行完成后，把本节复制到新的结果文档并填写，禁止只写“跑通了”：

```text
Experiment: B3-complex-v2 data anchor
Git commit:
Host / CPU:
Python:
Start UTC:
End UTC:
Exit code:
Output root:
Dataset ID:

Counts:
  train:
  val:
  test:

Replay / stems-sum / Ledger:
  train:
  val:
  test:

Split contract pass:
Split failed checks:

Quality gate:
  train pass / failed checks:
  val pass / failed checks:
  test pass / failed checks:

Human audit:
  reviewed:
  severe failures:
  template-level failures:
  decision:

Final decision: PASS / FAIL
Allowed next action:
Known limitations:
```
