# 下一步实验：真实单源 Catalog 锚点

更新日期：2026-08-13

## 1. 为什么现在必须做这一环

远端提交 `9d68604`、`2cd3e38` 已生成 5,000 条 B3-v2 合成混音并完成一次训练；远端报告的事件 F1 从较简单数据上的 0.948 降到较难数据上的 0.917，说明模板复杂度确实提高了。但 `docs/16_synthetic_audio_fundamental_issues.md` 的第三轮人工试听确认：合成 speech、vocal 和 SFX 只是正弦/噪声等占位波形，听觉内容并不支持 caption；固定短句也与具体 source 无关。

因此，现有 B3-v2 只能作为“输出格式、时间结构和训练代码可运行”的工程锚点，不能作为真实复杂音频语义能力的论文结果。下一步不应继续调 loss、RL 或 query 数量，而应先冻结下面这条真实数据链：

```text
真实单源音频 + 原始/人工逐源标注
  -> 严格 source catalog
  -> 文件解码/质量/许可/重复检查
  -> speaker/song/original-video/uploader 连通分组
  -> train/val/test 先分源、后混音
  -> 人工逐源试听 gate
  -> 100 条 real-mix smoke
  -> 人工混音试听 gate
  -> 才允许扩大数据和重新训练 B3
```

## 2. 已实现的代码

- `sceneledger-prepare-sources prepare`：校验 schema、解码音频、计算 SHA256、标准化后的 waveform fingerprint、duration、RMS 和 clipping 比例；按泄漏连通组切分并生成 `all/train/val/test.jsonl`、`source_audit.csv` 和机器可读报告。
- `sceneledger-prepare-sources validate-audit`：检查人工试听表完整性与通过率，并把审核结果通过 SHA256 绑定到四个 catalog 文件。
- `CatalogSourcePool`：只从冻结的单一 fold 读取真实 source；把真实 caption/transcript、identity、source group、dataset、license 和 annotation origin 写入 scene manifest 和 target Ledger。
- `sceneledger-render` 对 catalog 默认 fail-closed：没有通过的人工 source audit 或 fold 不匹配时拒绝渲染。
- sampler 在同一场景中不重复原始文件，并为多说话人/多歌手模板强制不同 identity。

由这些真实单源进行的程序混音在 provenance 中记为 **Level B**，不是 Level A：renderer 精确知道被注入来源、placement、stem 和时间，但程序混音不等于原生真实场景，模型/LLM 扩写的声学属性也不会因此升级成人工真值。

## 3. 原始 catalog schema

每行一个 JSON object，严格遵循 `sceneledger.source_catalog.v1`。可从 `configs/data/real_sources.example.jsonl` 复制字段，但示例路径本身不能直接运行。

必须字段：

| 字段 | 定义 |
|---|---|
| `source_id` | 全项目唯一、稳定的源片段 ID |
| `kind` | `speech/vocal/music/sfx/ambience` 之一 |
| `audio_path` | 相对 `--audio-root` 的路径；不得逃逸该根目录 |
| `source_group` | 最小不可拆分身份，如原录音、speaker 或 song |
| `leakage_groups` | 可选的额外泄漏身份，如 `speaker:*`、`song:*`、`video:*`、`uploader:*` |
| `caption` | 必须由该音频本身支持的逐源文本；speech 写真实 transcript，vocal 写可授权 lyric/描述，music/SFX 写可听属性 |
| `dataset` | 来源语料库名称 |
| `license` | 精确许可字符串；CLI 使用同一字符串做 allowlist |
| `annotation_origin` | `human/dataset/asr/audio_model/llm_rewrite` 之一 |

建议字段：`identity`、`language`、`attribution`、`original_url`。如果上游已有官方 split，写 `split=train/val/test`；脚本会尊重它，但同一泄漏连通组不得出现冲突 split。未知许可不要猜测，保留在隔离区而不是写进 catalog。

对于互联网音视频，至少设置：

```json
{
  "source_group": "video:BVID-or-platform-id",
  "leakage_groups": ["uploader:stable-id", "music_work:stable-id"],
  "original_url": "..."
}
```

若同一原视频切成多个片段，它们必须共享 `source_group`。同一说话人、歌曲或上传者跨多个文件时，使用 `leakage_groups` 把它们连接起来。脚本按这些 token 的连通分量整体切分；例如 A 与 B 共享 speaker、B 与 C 共享 video，则 A/B/C 不会被拆到不同 fold。

## 4. 服务器执行顺序

以下命令假设仓库位于 `$REPO`，音频根目录位于 `$AUDIO_ROOT`。先执行 editable install，避免 Python 导入服务器上另一个旧 checkout：

```bash
cd "$REPO"
python -m pip install -e '.[data,dev]'
python -c 'import sceneledger, pathlib; print(pathlib.Path(sceneledger.__file__).resolve())'
```

最后一条必须打印当前 `$REPO/src/sceneledger/...`。

### 4.1 准备原始 metadata

把各语料的官方 manifest 转成一个原始 `raw_sources.jsonl`。不要把 `configs/data/real_sources.example.jsonl` 当真实数据使用。第一批 pilot 建议每个 kind 至少 60 个独立泄漏组；默认 gate 只是运行下限：每个 fold、每个 kind 至少 4 个 record 和 4 个连通 group。

逐源 caption 的最低要求：

- speech：真实 transcript；speaker identity 可用语料 ID，不推断敏感属性；
- vocal：能确认的歌词或“不可辨歌词的某类演唱”描述；不要让 LLM 凭空补歌词；
- music：流派/乐器/节奏/情绪只写可人工复核属性；
- SFX/ambience：声源、材质、动作、持续/瞬态、距离只写可听证据；
- LLM 只能重写已有属性，`annotation_origin=llm_rewrite`，不能把推测升级为真值。

### 4.2 运行 source gate 与分组切分

先把确实允许本项目使用的许可逐项加入 allowlist；下面字符串仅为命令格式示例，不是法律判断：

```bash
sceneledger-prepare-sources prepare \
  --input "$META_ROOT/raw_sources.jsonl" \
  --audio-root "$AUDIO_ROOT" \
  --output-dir "$WORK_ROOT/source_catalogs_v1" \
  --allow-license 'CC0-1.0' \
  --allow-license 'CC BY 4.0' \
  --split-ratios 0.8,0.1,0.1 \
  --seed 20260813 \
  --audit-per-kind 12 \
  --min-records-per-kind-per-split 4 \
  --min-groups-per-kind-per-split 4 \
  --min-caption-unique-fraction 0.5
```

若退出码非零，只看 `source_catalog_report.json` 中失败的 checks 并修数据；不要降低门槛来“跑通”。重点检查：

- 所有音频可解码且 duration/RMS/clipping 合格；
- file hash 和标准化 waveform fingerprint 均不重复；
- 五种 kind 均存在；
- caption 不是少数模板句的复制；
- 三个 fold 的 group 和 content hash 完全不交叉；
- 每个 fold 有足够多独立 source group，能支持 3–4 speaker 等模板。

### 4.3 人工逐源试听

打开 `source_audit.csv`，按其中相对路径在 `$AUDIO_ROOT` 播放。每行填写：

- `audible_y_n`：目标声源是否清楚可听；
- `caption_correct_y_n`：文本是否由音频支持；
- `kind_correct_y_n`：speech/vocal/music/sfx/ambience 类别是否正确；
- `notes`：错误类型和应如何修正。

每个 kind 默认审核 12 条（最低 gate 为 10 条），三个 yes 指标均需至少 90%。修正 caption 后必须重新运行 `prepare`，因为 catalog hash 已变化；不要手工编辑生成后的 fold catalog。

```bash
sceneledger-prepare-sources validate-audit \
  --preparation-report "$WORK_ROOT/source_catalogs_v1/source_catalog_report.json" \
  --audit-csv "$WORK_ROOT/source_catalogs_v1/source_audit.csv" \
  --output "$WORK_ROOT/source_catalogs_v1/source_audit_report.json" \
  --min-per-kind 10 \
  --min-pass-rate 0.90
```

### 4.4 生成首批真实混音 smoke

复制 `configs/data/real_mix.example.yaml` 三次。每个配置必须分别设置：

- `catalog_path`：对应 fold 的 `train.jsonl/val.jsonl/test.jsonl`；
- `expected_split`：与文件一致；
- `audio_root`：原始音频根；
- `audit_report_path`：上一步通过的 `source_audit_report.json`；
- 三个 fold 使用不同 `seed_base/template_seed/scene_id_prefix`。

先只渲染 100/30/30，不要立即做 5k：

```bash
sceneledger-render --config configs/data/real_train.yaml --output-dir "$WORK_ROOT/real_mix_v1/train" --limit 100 --validate
sceneledger-render --config configs/data/real_val.yaml   --output-dir "$WORK_ROOT/real_mix_v1/val"   --limit 30  --validate
sceneledger-render --config configs/data/real_test.yaml  --output-dir "$WORK_ROOT/real_mix_v1/test"  --limit 30  --validate
```

随后复用 `sceneledger-preflight-data`、`sceneledger-human-audit` 和 `sceneledger-validate-experiment-data` 做 scene complexity、混音试听和最终 artifact binding。真实 smoke 必须额外人工确认：speech/lyrics/SFX 确实可听、caption 与源一致、多个 speaker 真的是不同 identity、音乐/歌词 stem 同歌时 source grouping 正确、RIR/echo/ducking 没有掩蔽目标。

## 5. Go / No-Go

只有下面条件全部满足，才进入“real-mix 1k → B3 重训”：

1. source preparation report `pass=true`；
2. source audit report `pass=true` 且 catalog hash 未变化；
3. 100/30/30 全部 deterministic replay、stem-sum、Ledger validation 通过；
4. real-mix 人工试听通过，特别是语义—波形一致性；
5. train/val/test 按全部 leakage identity 无交集；
6. 模型训练配置记录这批 catalog 和 mixture manifest 的 SHA256。

任一失败时停在数据层修复。现阶段不要做 RL/DPO、LLM judge、query architecture 或大规模网络视频伪标注，因为这些模块无法补救错误的源语义真值。

## 6. 本地已验证范围与仍需服务器完成的工作

本地单元测试已覆盖：许可 fail-closed、音频 probe、重复 fingerprint、分组切分、官方 fixed split、二级 leakage identity、人工审核与 catalog hash 绑定、篡改拒绝、真实 metadata 进入 Ledger、不同说话人 identity 采样。

本机没有真实语料和算力，因此尚未完成：各数据集专用 metadata converter、真实 source 人工试听、100/30/30 real-mix、真实混音质量验收和重训练。专用 converter 应在拿到服务器目录与官方 manifest 的 3–5 行真实样例后实现；在此之前写死目录结构会制造另一个不可验证假设。
