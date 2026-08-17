# LLM 具体音源选择与时间规划实验

更新日期：2026-08-16

## 1. 本轮回答的问题

本轮不再只让 LLM 选择 `dog_bark`、`rain` 之类的类别，而是允许它在冻结候选集中决定：

1. 每个模板 slot 使用哪一个具体的、经过 source catalog 审计的音频；
2. 每个音频在 0.1 秒网格上的 onset；
3. 由这些 onset 和源时长形成怎样的先后、同时和重叠关系。

LLM 不能提交任意文件路径。它只能返回任务中出现的 `catalog_source_id`。caption、transcript、lyrics、source kind、source group、文件哈希和时长仍来自 catalog，事件活动区间仍在渲染后由持久化 stem 计算。这样既允许 LLM 真正改变混音内容与时间结构，又不会把 LLM 的文本猜测当成声学真值。

## 2. 新增制品和数据流

```text
passed source catalogs
        |
        v
label_inventory.json -- catalog paths + SHA256
        |
        v
source_timeline_tasks.jsonl
  - fixed template / seed / 12.0 s duration
  - exact candidate source slate per slot
  - audited caption, label, duration, split, group and file hash
        |                                  |
        |                                  +--> deterministic rule selection
        |                                       rule_recipes.jsonl
        v
LLM chooses exact source IDs + onset_sec
        |
        v
strict compiler / allowlist / duration / leakage validation
        |
        v
llm_recipes.jsonl
        |
        v
same deterministic renderer -> mixture + processed stems + Ledger
```

`sceneledger.scene_recipe.v2` 中新增：

```json
{
  "scene_duration_sec": 12.0,
  "source_plan": [
    {
      "slot_id": "speech_1",
      "kind": "speech",
      "catalog_source_id": "librispeech:84-121123-0001",
      "onset_sec": 1.2
    },
    {
      "slot_id": "sfx_1",
      "kind": "sfx",
      "catalog_source_id": "fsd50k:12345",
      "onset_sec": 4.7
    }
  ]
}
```

这里的 `source_plan` 是不可变实验输入。renderer 会在模板自带调度之后再次强制应用它；如果实际抽到的 source 或 onset 与计划不一致，渲染立即失败，而不是只把 LLM 输出写进 metadata。

## 3. 当前支持范围

第一版支持：

- `speech_with_sfx`
- `speech_ambience_sfx`
- `speech_over_music`
- `music_with_sfx`
- `speech_music_sfx`
- `ambient_with_intermittent_sfx`
- `overlapping_speakers`

第一轮建议只运行前三个已经具备可靠 source bank 的模板，优先从 `speech_with_sfx` 和 `speech_ambience_sfx` 开始。

暂不支持 `lyrics_over_music`、`rich_band`、`repeated_event` 和 `multi_speaker_ambient_events` 的 LLM exact-source 规划。前两类要求 music/vocal 必须来自同一 stem group；后两类分别要求多次重复和同一 speaker 的 persistent track。这些约束不能用“任意不重复 source”规则处理，应在第一轮验证通过后实现 bundle/group planner。

## 4. 前置条件

不得使用 `real_mix_v7` 的占位 source 和 MOSS 伪 caption。输入 catalog 必须满足：

- 已按 train/val/test 冻结，当前 inventory 必须只包含一个且非空的 split；
- source audit 报告通过；
- 每条候选有 `source_id`、`source_group`、caption、duration、split；
- 正式 renderer 使用的 audited catalog 与 inventory 中的 catalog SHA256 完全一致；
- speech/SFX 不依靠裁剪来塞入 scene，否则完整 caption 会失去证据；
- 不同 source ID 之间不存在重复 content hash 或 leakage group。

`make_recipe_mix_config.py` 会对 source-timeline recipe 增加 catalog hash 对齐检查，避免用 catalog A 生成任务却用 catalog B 渲染。

## 5. 生成冻结任务与规则对照

下面假设服务器已经完成 LibriSpeech、ESC-50 和 FSD50K 的 train split 准备及人工 source audit。

```bash
export REPO=/path/to/complex-audio-caption
export RUN_ROOT=/data/sceneledger_runs/llm_source_timeline_v1
export LIBRI_TRAIN=/path/to/librispeech/prepared/train.jsonl
export ESC_TRAIN=/path/to/esc50/prepared/train.jsonl
export FSD_TRAIN=/path/to/fsd50k/prepared/train.jsonl

cd "$REPO"
python -m pip install -e '.[data,dev]'
mkdir -p "$RUN_ROOT/recipes"

sceneledger-recipes inventory \
  --catalog "$LIBRI_TRAIN" \
  --catalog "$ESC_TRAIN" \
  --catalog "$FSD_TRAIN" \
  --output "$RUN_ROOT/recipes/label_inventory.json"

sceneledger-recipes llm-source-timeline-tasks \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --count 120 --seed 20260816 \
  --candidates-per-slot 12 --scene-duration-sec 12.0 \
  --template-weight speech_with_sfx=1 \
  --template-weight speech_ambience_sfx=1 \
  --output "$RUN_ROOT/recipes/source_timeline_tasks.jsonl"

sceneledger-recipes rule-source-timeline \
  --tasks "$RUN_ROOT/recipes/source_timeline_tasks.jsonl" \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --strategy keyword \
  --output "$RUN_ROOT/recipes/rule_recipes.jsonl" \
  --report "$RUN_ROOT/recipes/rule_validation.json"
```

任务中的每个 slot 最多展示 12 个具体候选。候选通过 dataset/primary-label round-robin 取样，避免一个大数据集或头部类别占满上下文。task 保存 inventory hash 和自身 SHA256；修改候选、prompt 或时间限制后，旧响应不能继续使用。

## 6. 调用 LLM

先用五条任务检查 endpoint 和 JSON 格式：

```bash
export LLM_API_KEY='只写入当前 shell，不写入仓库或配置'

python scripts/call_recipe_llm.py \
  --tasks "$RUN_ROOT/recipes/source_timeline_tasks.jsonl" \
  --output "$RUN_ROOT/recipes/llm_responses.jsonl" \
  --endpoint https://YOUR-ENDPOINT/v1/chat/completions \
  --model YOUR_MODEL \
  --api-key-env LLM_API_KEY \
  --json-mode --temperature 0.2 --limit 5
```

检查前五行后去掉 `--limit` 重跑。同一个输出文件会按 `task_id` 断点续跑。每行 response 会记录 model、temperature、JSON mode 和 task hash，但不记录 API key。

LLM 必须返回：

```json
{
  "context": "street",
  "difficulty": "hard",
  "source_plan": [
    {
      "slot_id": "speech_1",
      "kind": "speech",
      "catalog_source_id": "候选集中的精确ID",
      "onset_sec": 0.8
    },
    {
      "slot_id": "sfx_1",
      "kind": "sfx",
      "catalog_source_id": "候选集中的精确ID",
      "onset_sec": 3.1
    }
  ],
  "relations": ["sequential"],
  "rationale": "The foreground event follows the utterance in a plausible street scene."
}
```

编译完整响应：

```bash
sceneledger-recipes compile-llm-source-timeline \
  --tasks "$RUN_ROOT/recipes/source_timeline_tasks.jsonl" \
  --responses "$RUN_ROOT/recipes/llm_responses.jsonl" \
  --model-name YOUR_MODEL \
  --inventory "$RUN_ROOT/recipes/label_inventory.json" \
  --output "$RUN_ROOT/recipes/llm_recipes.jsonl" \
  --report "$RUN_ROOT/recipes/llm_validation.json"

sceneledger-recipes compare \
  --left "$RUN_ROOT/recipes/rule_recipes.jsonl" \
  --right "$RUN_ROOT/recipes/llm_recipes.jsonl" \
  --output "$RUN_ROOT/recipes/rule_vs_llm.json"
```

`compare` 必须确认 scene 数、seed、template、scene duration 和每条任务的 candidate-task hash 完全相同。

## 7. 编译器拒绝条件

以下任一情况都会使整组 compile 失败：

- 缺失、额外或重复 task ID；
- task/inventory hash 改变；
- response 缺少调用 provenance，或 LLM model / response task hash 与命令参数不一致；
- 缺 slot、重复 slot 或 kind 不一致；
- source ID 不在该 slot 的冻结候选中；
- 同一 scene 重复 source ID、source group 或 leakage group；
- onset 不是 0.1 秒网格；
- onset 超过该候选的 `max_onset_sec`；
- speech、vocal 或 SFX 在指定 onset 后无法完整放入 scene；
- candidate 中的 caption、duration、split、group、label、dataset 或 file hash 与 catalog 不一致；
- `overlapping_speakers` 的两个 speech 实际重叠少于 0.1 秒。

不要捕获这些错误后自动换一个随机 source。正确做法是保留失败响应，修正 prompt/模型或生成一个具有新 hash 的新任务版本。

## 8. 渲染 matched Rule/LLM 两组

base config 必须使用与 inventory 完全相同的 catalog，并允许 12 秒 duration：

```yaml
sampler:
  duration_range: [12.0, 12.0]
  stable_unique_source_ids: true
  random_crop_backgrounds: true
  loop_background_to_scene: true
```

生成并运行两组：

```bash
for arm in rule llm; do
  python scripts/make_recipe_mix_config.py \
    --base-config "$RUN_ROOT/base_train_mix.yaml" \
    --recipes "$RUN_ROOT/recipes/${arm}_recipes.jsonl" \
    --inventory "$RUN_ROOT/recipes/label_inventory.json" \
    --scene-id-prefix "source_timeline_${arm}" \
    --output "$RUN_ROOT/recipes/${arm}_mix.yaml"

  bash scripts/run_recipe_mix_arm.sh \
    "$RUN_ROOT/recipes/${arm}_mix.yaml" \
    "$RUN_ROOT/rendered/${arm}" recipe_scale
done
```

不要让 LLM 组和 Rule 组共享输出目录，也不要覆盖失败运行。renderer 输出的 scene manifest 会同时保存计划 source ID、计划 onset、实际 source path、source hash、processed stem 和 Ledger，因此可以逐层定位问题。

## 9. 第一轮评价与 Go/No-Go

先评价数据，不训练 caption 模型。两组各 120 条，使用隐藏 arm 名称的 paired blind review，至少检查：

- scene plausibility：这些具体声音是否合理共现；
- temporal plausibility：顺序和重叠是否合理；
- mixture naturalness；
- 每个计划 source 是否实际可听；
- source caption 是否与 isolated stem 一致；
- mixture 中的 event/span 是否与 stem evidence 一致；
- source/label/dataset 覆盖以及单个 source 最大复用次数。

进入扩大实验的条件：

1. 两组 schema、replay、stem-sum、source identity 和 temporal evidence gate 全部通过；
2. LLM 没有任何被编译器接受的 invented source；
3. LLM 组的 source audibility、caption accuracy 和 timestamp alignment 不低于 Rule 组；
4. paired blind review 显示 LLM 在 scene/temporal plausibility 上有稳定优势；
5. 优势不是由少量 source 被高频重复造成。

若 LLM 只让 rationale 更好看，但盲听没有提高，则结论是“不值得扩大”，继续使用规则 sampler。只有该数据实验通过，才生成相同规模的训练集并在冻结 source-disjoint test 上比较下游 caption 模型。

匿名音频/stem 打包、双 reviewer 表格、制品防篡改、解盲统计和默认
Go/No-Go 阈值已经实现为 `sceneledger-mixture-review`。完整命令和评分说明见
`docs/40_v8_forensics_and_paired_llm_mix_review.md`。不要手动按照目录名进行非盲
比较，也不要在 review 完成前打开 private assignment key。

## 10. 后续扩展顺序

第一轮通过后按以下顺序推进：

1. `paired_stem_bundle`：LLM 选择同一 song group 的 music/vocal stem bundle；
2. `persistent_track_bundle`：LLM 选择同一 speaker 的多个 utterance，并规划 turn-taking；
3. `event_instance_plan`：同一 source 的重复、响应和 interruption；
4. audio-capable LLM 作为渲染后候选重排器，但不能替代 stem 和人工真值；
5. 以真实互联网音视频的统计分布约束模板、source 数、overlap 和 acoustic parameter 的采样先验。
