# 20 小时数据扩充与 Rule/LLM 混音实验

更新日期：2026-08-14

> 更新：本文的 LLM arm 只选择类别标签。允许 LLM 从冻结候选中选择具体
> 音频并规划 0.1 秒 onset 的新版实验见
> `docs/39_llm_source_and_timeline_planner.md`；数据与 source-audit 前置门禁保持不变。

## 1. 本轮只回答两个问题

20 小时不能同时证明数据、mixer、LLM、训练和模型架构都有效。本轮按顺序回答：

1. **数据锚点问题**：LibriSpeech + ESC-50 + FSD50K 能否形成可追溯、可听、具有足够类别多样性的 60 条复杂混音？
2. **组合策略问题**：锚点通过后，在模板数、随机种子和声学参数相同的条件下，LLM 选择的“场景—音效类别”组合是否比均匀随机组合更合理，并且至少不弱于关键词规则？

LLM 在本轮只提出 `scene recipe`，即从已审计标签集合中选择哪些类别适合共同出现。它不能产生或修改 source ID、音频、speech transcript、lyrics、source 真值、人物身份、时间戳、stem、Ledger 或最终 caption。因此 LLM 幻觉最多造成一条“组合不合理”的 recipe，不能伪造源级标注。所有 recipe 均须通过 schema、inventory 和人工合理性检查。

## 2. 当前分支上的诊断结论

远端 `main` 新增的 `real_mix_v6` 使用 GTZAN 的 classical/jazz/blues 作为“纯器乐”候选。该制品可作为诊断样例，但**不能作为论文数据锚点**：

- genre 标签不能证明片段没有 singing 或 speech；
- GTZAN 没有 vocal-absence/stem 监督；
- 当前获取路径和逐文件许可没有进入 source-bank 审计；
- MOSS source caption 是伪标签，不是 waveform ground truth；
- `present_in_mix` 不能替代 isolated-stem 可听性、相对遮蔽和人工听审。

所以本轮不继续调 v6 的 gain/ducking，也不基于 v6 训练。music/lyrics 等通过 Slakh/MUSDB/精确歌词数据门禁后再单独推进。

## 3. 本轮新增的可执行能力

### 3.1 UrbanSound8K 严格前景子集

`scripts/download_urbansound8k.py` 从官方 Zenodo record 下载约 6 GB 的 v1 archive，固定发布方 MD5，支持断点续传、安全解压和完整的 8732 WAV 检查。

`sceneledger-prepare-urbansound8k` 只保留 `salience=1` 的六种离散前景类：`car_horn`、`dog_bark`、`drilling`、`gun_shot`、`jackhammer`、`siren`。它排除 continuous bed、多人/多声源 `children_playing` 和可能含人声的 `street_music`。同一 Freesound `fsID` 被绑定为同一 leakage group。该数据是 CC BY-NC 3.0，只能进入 non-commercial research profile。

UrbanSound8K 是 **P1b 可选扩展**，不是 P1 替代品。它只有六类严格 SFX，不能替代 FSD50K 的类别覆盖，也不能提供 ambience。

### 3.2 可审计的 scene recipe

| 制品 | 作用 | 边界 |
|---|---|---|
| `label_inventory.json` | frozen catalog hash、标签和数量 | 不含 LLM 猜测 |
| `uniform_recipes.jsonl` | 场景无关、均匀选标签的对照 | 不使用语义知识 |
| `keyword_recipes.jsonl` | 关键词匹配 context 与标签 | 不调用外部模型 |
| `llm_tasks.jsonl` | 固定 template、seed、允许标签 | 不允许模型造标签 |
| `llm_recipes.jsonl` | 编译验证后的 LLM 方案 | 不允许 source/caption/timestamp |
| `*_review.csv` | 人工检查组合合理性 | 不替代混音听审 |

renderer 将 recipe plan 与 inventory 的 SHA256 写入 scene metadata 和 data card。recipe 指定标签会从所有 catalog 的全量精确标签索引中抽取，不经过通用候选条数截断。

## 4. 三组 matched 对照

固定 120 个 scene、相同 template 序列、逐 scene seed、catalog、bank weight、duration、RMS target、SNR、RIR/echo、renderer、自动门禁和人工抽样数。只改变 recipe proposal：

| Arm | 策略 | 科研问题 |
|---|---|---|
| U: uniform | 标签均匀抽取，与 context 无关 | 无组合先验的下界 |
| R: keyword | 冻结关键词表选择兼容标签 | 便宜、确定性的强规则基线 |
| L: LLM | 只从允许标签选取并给理由 | 是否带来超出关键词的组合知识 |

首轮只用 `speech_with_sfx` 和 `speech_ambience_sfx`，各约一半。不要同时加入 music、vocal 或 overlapping speakers。

主要指标依次是：recipe 合法率；人工 `plausible_y_n`/`label_compatible_y_n`；唯一组合、标签和 context 覆盖；renderer 自动门禁；分层混音听审的 event audibility、caption accuracy、timestamp alignment 与 overall decision。不能把“rationale 看起来更自然”当作结果，也不能看过结果后改变阈值。

## 5. 20 小时时间线与停止条件

| 时间 | 必做工作 | 进入下一阶段的条件 |
|---|---|---|
| H0–H0.5 | 环境、测试、policy 预检 | 全 pass |
| H0.5–H6 | FSD50K full 下载/prepare | checksum 和完整性 pass |
| H6–H9 | 30 SFX + 30 ambience 源听审 | 三项通过率均不低于 90% |
| H9–H12 | 60 条 ESC+FSD pilot，全量听审 | 0 severe；fail 不超过 2/60 |
| H12–H13 | 生成 U/R/L recipe 与 review | schedule matched；无非法标签 |
| H13–H17 | 三组各渲染 120 条 | 三组自动 gate 均 pass |
| H17–H20 | 每组分层听审最多 40 条并汇总 | 得到可解释结论，不强行 GO |

若下载慢，时间线自动后移。**H12 时 P1 未 GO，剩余时间全部用于完成/诊断 P1，不运行 LLM，不渲染 recipe arms。** 若 P1 在 H10 前 GO 且带宽充足，可下载 UrbanSound8K，但作为独立 P1b，不能悄悄替换三组主实验的 catalog。

若三组 120 条在 H17 前完成并通过自动门禁，可把三组 `--count` 同时扩为 300，在全新输出目录重跑；不能只扩大表现最好的一组。人工仍采用冻结的分层抽样。

## 6. H0：环境和策略预检

```bash
export REPO=/path/to/complex-audio-caption
export DATA_ROOT=/data/sceneledger_data
export RUN_ROOT=/data/sceneledger_runs/twenty_hour_recipe_v1
export LIBRI_PREP=/path/to/passed/librispeech/prepared
export ESC_PREP=/path/to/passed/esc50/prepared
export LIBRI_ROOT=/data/sceneledger_data/librispeech
export ESC_AUDIO=/data/sceneledger_data/esc50/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio

cd "$REPO"
git pull --ff-only
python -m pip install -e '.[data,dev]'
python -m pytest -q
command -v ffmpeg
command -v zip
python -c 'import sceneledger, pathlib; print(pathlib.Path(sceneledger.__file__).resolve())'
mkdir -p "$RUN_ROOT"

sceneledger-validate-source-policy \
  --policy configs/data/source_bank_policy.yaml \
  --profile d1_urban_sfx_research \
  --output "$RUN_ROOT/source_policy_precheck.json"
```

import path 必须位于当前 `$REPO/src/sceneledger`。测试或 policy 失败时先停止。

## 7. P1：先完成 FSD50K 数据锚点

完整说明见 `docs/29_expanded_source_bank_protocol.md` 第 4 节。最短路径：

```bash
python scripts/download_fsd50k.py \
  --output-dir "$DATA_ROOT/fsd50k" --profile full

export FSD_ROOT="$DATA_ROOT/fsd50k/FSD50K"
export FSD_PREP="$RUN_ROOT/sources/fsd50k/prepared"

sceneledger-prepare-fsd50k \
  --root "$FSD_ROOT" \
  --output-dir "$RUN_ROOT/sources/fsd50k" \
  --allow-license CC0-1.0 \
  --allow-license 'CC BY 3.0' \
  --audit-per-kind 30 --min-per-kind-per-split 50
```

人工填完 `$FSD_PREP/source_audit.csv` 后：

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$FSD_PREP/source_catalog_report.json" \
  --audit-csv "$FSD_PREP/source_audit.csv" \
  --output "$FSD_PREP/source_audit_report.json" \
  --min-per-kind 30 --min-pass-rate 0.90 \
  --required-split test --min-per-kind-per-required-split 10

python scripts/make_real_speech_sfx_pilot_config.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$LIBRI_PREP" \
  --esc50-audio-root "$ESC_AUDIO" \
  --esc50-prepared "$ESC_PREP" \
  --fsd50k-root "$FSD_ROOT" \
  --fsd50k-prepared "$FSD_PREP" \
  --output "$RUN_ROOT/expanded_pilot_test.yaml"

bash scripts/run_real_speech_sfx_pilot.sh \
  "$RUN_ROOT/expanded_pilot_test.yaml" \
  "$RUN_ROOT/expanded_pilot_output" expanded_pilot
```

自动门禁通过后听完 60 条 `human_audit_tasks.csv`，再运行：

```bash
bash scripts/summarize_real_speech_sfx_pilot.sh \
  "$RUN_ROOT/expanded_pilot_output"
```

P1 GO：source audit 各项至少 90%，mixture `max_severe=0`，overall failure 不超过 2/60，无系统性缺声、错 caption 或不可听问题。

## 8. P1b：可选 UrbanSound8K 第三 SFX bank

仅在 P1 已 GO，或下载不争用 FSD 关键带宽时执行：

```bash
python scripts/download_urbansound8k.py \
  --output-dir "$DATA_ROOT/urbansound8k"

export US8K_ROOT="$DATA_ROOT/urbansound8k/UrbanSound8K"
export US8K_PREP="$RUN_ROOT/sources/urbansound8k/prepared"

sceneledger-prepare-urbansound8k \
  --root "$US8K_ROOT" \
  --output-dir "$RUN_ROOT/sources/urbansound8k" \
  --audit-per-kind 30 --min-per-split 50
```

听完 `$US8K_PREP/source_audit.csv` 的 30 条 SFX 后：

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$US8K_PREP/source_catalog_report.json" \
  --audit-csv "$US8K_PREP/source_audit.csv" \
  --output "$US8K_PREP/source_audit_report.json" \
  --min-per-kind 30 --min-pass-rate 0.90 \
  --required-split test --min-per-kind-per-required-split 10

sceneledger-validate-source-policy \
  --policy configs/data/source_bank_policy.yaml \
  --profile d1_urban_sfx_research \
  --catalog librispeech="$LIBRI_PREP/test.jsonl" \
  --catalog esc50="$ESC_PREP/test.jsonl" \
  --catalog fsd50k="$FSD_PREP/test.jsonl" \
  --catalog urbansound8k="$US8K_PREP/test.jsonl" \
  --require-catalogs \
  --output "$RUN_ROOT/urban_policy_with_catalogs.json"
```

制作含第三 bank 的独立配置：

```bash
python scripts/make_real_speech_sfx_pilot_config.py \
  --librispeech-root "$LIBRI_ROOT" \
  --librispeech-prepared "$LIBRI_PREP" \
  --esc50-audio-root "$ESC_AUDIO" \
  --esc50-prepared "$ESC_PREP" \
  --fsd50k-root "$FSD_ROOT" \
  --fsd50k-prepared "$FSD_PREP" \
  --urbansound8k-root "$US8K_ROOT" \
  --urbansound8k-prepared "$US8K_PREP" \
  --output "$RUN_ROOT/expanded_pilot_with_urban.yaml"
```

首轮 U/R/L 建议仍使用已 GO 的 P1 两库配置，Urban 做独立消融。若决定以三库做主实验，三组必须全部使用相同三库 base config。

## 9. 生成 matched U/R/L recipe

下例使用 LibriSpeech、ESC-50、FSD50K 的 test catalog；若以 P1b 为主实验，再给 inventory 增加一行 Urban catalog。

```bash
mkdir -p "$RUN_ROOT/recipes"

sceneledger-recipes inventory \
  --catalog "$LIBRI_PREP/test.jsonl" \
  --catalog "$ESC_PREP/test.jsonl" \
  --catalog "$FSD_PREP/test.jsonl" \
  --output "$RUN_ROOT/recipes/label_inventory.json"

sceneledger-recipes rules \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --count 120 --seed 20260814 --strategy uniform --prefix uniform \
  --template-weight speech_with_sfx=1 \
  --template-weight speech_ambience_sfx=1 \
  --output "$RUN_ROOT/recipes/uniform_recipes.jsonl" \
  --report "$RUN_ROOT/recipes/uniform_validation.json"

sceneledger-recipes rules \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --count 120 --seed 20260814 --strategy keyword --prefix keyword \
  --template-weight speech_with_sfx=1 \
  --template-weight speech_ambience_sfx=1 \
  --output "$RUN_ROOT/recipes/keyword_recipes.jsonl" \
  --report "$RUN_ROOT/recipes/keyword_validation.json"

sceneledger-recipes llm-tasks \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --count 120 --seed 20260814 --max-labels-per-kind 500 \
  --template-weight speech_with_sfx=1 \
  --template-weight speech_ambience_sfx=1 \
  --output "$RUN_ROOT/recipes/llm_tasks.jsonl"
```

本地 OpenAI-compatible endpoint：

```bash
python scripts/call_recipe_llm.py \
  --tasks "$RUN_ROOT/recipes/llm_tasks.jsonl" \
  --output "$RUN_ROOT/recipes/llm_responses.jsonl" \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model YOUR_LOCAL_INSTRUCT_MODEL \
  --no-auth --json-mode --temperature 0.4
```

hosted endpoint 的 key 只放环境变量，不能写入 YAML/JSONL：

```bash
export LLM_API_KEY='set-this-in-the-shell-only'
python scripts/call_recipe_llm.py \
  --tasks "$RUN_ROOT/recipes/llm_tasks.jsonl" \
  --output "$RUN_ROOT/recipes/llm_responses.jsonl" \
  --endpoint https://YOUR-ENDPOINT/v1/chat/completions \
  --model YOUR_MODEL --api-key-env LLM_API_KEY \
  --json-mode --temperature 0.4
```

脚本逐条 checkpoint，可安全恢复。可先用 `--limit 5` 测试连通性；正式运行时去掉 limit，最终 compile 会拒绝缺失任务。

```bash
sceneledger-recipes compile-llm \
  --tasks "$RUN_ROOT/recipes/llm_tasks.jsonl" \
  --responses "$RUN_ROOT/recipes/llm_responses.jsonl" \
  --model-name YOUR_MODEL \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --output "$RUN_ROOT/recipes/llm_recipes.jsonl" \
  --report "$RUN_ROOT/recipes/llm_validation.json"

sceneledger-recipes compare \
  --left "$RUN_ROOT/recipes/uniform_recipes.jsonl" \
  --right "$RUN_ROOT/recipes/keyword_recipes.jsonl" \
  --output "$RUN_ROOT/recipes/uniform_vs_keyword.json"
sceneledger-recipes compare \
  --left "$RUN_ROOT/recipes/uniform_recipes.jsonl" \
  --right "$RUN_ROOT/recipes/llm_recipes.jsonl" \
  --output "$RUN_ROOT/recipes/uniform_vs_llm.json"

for arm in uniform keyword llm; do
  sceneledger-recipes review-sheet \
    --recipes "$RUN_ROOT/recipes/${arm}_recipes.jsonl" \
    --output "$RUN_ROOT/recipes/${arm}_review.csv"
done
```

`compare` 必须报告 count、seed、template 全部 matched。人工至少每模板审 15 条/arm；若时间允许可审完全部 recipe，此步骤不播放音频。invented label 会在 compile 阶段直接失败。

## 10. 渲染三个 recipe arm

以通过 P1 的 config 为 base：

```bash
for arm in uniform keyword llm; do
  python scripts/make_recipe_mix_config.py \
    --base-config "$RUN_ROOT/expanded_pilot_test.yaml" \
    --recipes "$RUN_ROOT/recipes/${arm}_recipes.jsonl" \
    --inventory "$RUN_ROOT/recipes/label_inventory.json" \
    --scene-id-prefix "recipe_${arm}" \
    --output "$RUN_ROOT/recipes/${arm}_mix.yaml"

  bash scripts/run_recipe_mix_arm.sh \
    "$RUN_ROOT/recipes/${arm}_mix.yaml" \
    "$RUN_ROOT/recipe_outputs/${arm}" recipe_scale
done
```

runner 会拒绝复用旧目录，记录 commit/Python/environment，执行 preflight、render、replay、stem-sum、Ledger 和 active-RMS 门禁，并生成分层听审表。自动门禁失败时保留报告，不要调参后覆盖旧目录。

填完每组 `human_audit_tasks.csv` 后：

```bash
for arm in uniform keyword llm; do
  sceneledger-human-audit summarize \
    --review-csv "$RUN_ROOT/recipe_outputs/${arm}/human_audit_tasks.csv" \
    --metadata "$RUN_ROOT/recipe_outputs/${arm}/human_audit_tasks.meta.json" \
    --output "$RUN_ROOT/recipe_outputs/${arm}/human_audit_summary.json" \
    --max-severe 0 --max-total-failures 4 \
    --template-failure-threshold 3
done
```

recipe-scale 是三组探索性比较，上述 `4` 是执行前冻结阈值。论文还需报告每项失败率和置信区间，不能只报 pass/fail。

## 11. 失败后只修责任环节

| 失败位置 | 允许动作 | 禁止动作 |
|---|---|---|
| FSD source audit | 收紧类别/PP 过滤，重新 prepare/audit | 改 mixer gain |
| source active RMS | 查解码、activity/RMS 计算 | 让 LLM 改 caption |
| recipe compile | 修 endpoint JSON 或 contract | 手工造不存在标签 |
| recipe plausible | 如实记录 U/R/L 失败率 | 删除失败行只报成功行 |
| replay/stem gate | 修 renderer，新目录重跑三组 | 只重跑 LLM 组 |
| caption/timestamp | 回看 source、stem、Ledger 定责 | 直接开始模型训练 |

修复产生新目录如 `recipe_v2`，旧报告不可覆盖。

## 12. 本轮不建议下载的内容

- **Slakh2100-redux**：archive 约 105 GB，解压后接近 500 GB；没有至少 650 GB 空间、稳定高速网络且 P1 未完成时，不进入关键路径。
- **Common Voice**：须经 Mozilla Data Collective 获取，适合后续 speech 多样性单变量消融。
- **MUSDB18-HQ**：须从官方入口取得；只能监督 vocal presence，不能提供逐字 lyrics。
- **Bilibili/Instagram/TikTok 爬取数据**：目前无可靠时间戳、许可和 source truth，只能进入 rights/privacy 审核后的真实域评估/弱监督，不能作 dry-source bank。

空闲带宽优先级为 FSD50K、UrbanSound8K，最后才是在另一存储卷后台下载 Slakh。数据量不是验收标准；通过门禁的 source、类别和 group 数才是。

## 13. 回传的最小制品

不要回传受限制的原始音频。保存并回传：

- commit、`pip_freeze.txt`、Python/import path；
- source policy、FSD/Urban preparation report、完成的 audit CSV/report；
- P1 config、preflight、manifest、quality report、human summary；
- inventory、三组 recipes、LLM tasks/responses、validation、matched comparison 和 recipe review；
- 三组 mix config、manifest、quality report、human summary；
- 每类失败各 3–5 个 sample ID、mixture/stem 路径和中文听感；
- 实际 wall-clock、下载大小、峰值磁盘占用和重试次数。

拿到这些制品后才能决定扩大到 300/arm、引入 UrbanSound、开始 Slakh music pilot，还是修复确定的数据/renderer 问题。没有 P1 GO 时，不以训练 loss 解释任何结果。
