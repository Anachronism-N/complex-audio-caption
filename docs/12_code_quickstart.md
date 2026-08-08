# 代码快速开始与服务器运行手册

本文对应首个可运行研究里程碑：统一数据契约、数据下载/整理、TAC-mini、Exact-CARC、MOSS SFT 转换、Track–Event slots 和分解式评价。

## 1. 已实现范围

```text
authorized/public audio
→ source manifest + group split
→ TAC-mini / Exact-CARC
→ Track–Event ledger + tagged caption
→ MOSS official SFT JSONL
→ MOSS encoder features
→ Track–Event slot structural training
→ event/time/text/track metrics
```

CPU 可测试部分不下载任何模型；MOSS 和 SAM Audio 均通过官方仓库延迟接入，没有复制上游代码或权重。

当前 slot checkpoint 只训练 source/event 结构、activity 和 pointer，不包含最终 event text decoder。论文完整模型应在结构稳定后，把 `local_feature` 投影为 MOSS/Qwen3 的 evidence prefix，并加入 event text loss。

## 2. CPU 环境

基础工具兼容 Python 3.9+；MOSS 服务器按官方建议使用 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,audio,download]"
pytest
```

`sceneledger[audio]` 为 FLAC/MP3 等格式加入 SoundFile；不安装时仍可通过 SciPy 处理 WAV。

## 3. 不下载数据的完整 smoke test

```bash
python scripts/make_toy_sources.py --output data/interim/toy

sceneledger validate-sources data/interim/toy/sources.jsonl

sceneledger render \
  --sources data/interim/toy/sources.jsonl \
  --config configs/data/tac_smoke.yaml \
  --output data/derived/tac_smoke

sceneledger validate-render data/derived/tac_smoke

sceneledger carc \
  --backgrounds data/interim/toy/sources.jsonl \
  --sources data/interim/toy/sources.jsonl \
  --config configs/data/exact_carc_smoke.yaml \
  --output data/derived/carc_smoke

sceneledger moss-sft \
  --ledgers data/derived/tac_smoke/ledgers.jsonl \
  --render-manifest data/derived/tac_smoke/render_manifest.jsonl \
  --output data/derived/tac_smoke/moss_train.jsonl
```

`validate-render` 检查 ledger、音频长度以及保存后的 stems 是否能重构 mixture。真实数据扩容前必须运行。

## 4. 下载公开数据

数据注册表在 `configs/data/datasets.yaml`。脚本不会默认同意任何许可：

```bash
sceneledger download \
  --registry configs/data/datasets.yaml \
  --output-root /data/sceneledger/raw \
  --dataset rirs_noises \
  --accept-license rirs_noises \
  --extract
```

可重复使用 `--dataset` 和 `--accept-license`。下载支持 `.part` 断点续传、可选 SHA-256 和防路径穿越解压。FSD50K、MUSDB18-HQ、Clotho 与私有 web 数据保留为 manual entry，因为它们需要逐版本/逐音频检查许可或平台条款。

## 5. 数据目录和 metadata

推荐把可训练单源整理为：

```text
/data/sceneledger/interim/sources/
  speech/
  lys/
  music/
  sfx/
  ambience/
```

复制 `configs/data/metadata_template.csv`，为文件提供文本和泄漏分组：

```bash
sceneledger organize \
  --input-root /data/sceneledger/interim/sources \
  --metadata /data/sceneledger/metadata.csv \
  --output /data/sceneledger/manifests/sources.jsonl \
  --assign-splits

sceneledger validate-sources \
  /data/sceneledger/manifests/sources.jsonl \
  --verify-hashes
```

`group_id` 的优先级应为 speaker/singer、音乐作品、原视频或录音 session，而不是单个切片。相同 `group_id` 永远进入同一 split。

互联网音视频 metadata 至少保留：

- 平台 ID 和时间段；
- uploader hash；
- 音频作品/work ID；
- 许可审核状态；
- 删除索引；
- 音频 SHA-256。

仓库不会提供绕过平台访问控制或批量抓取 Bilibili/Instagram/TikTok 的代码。已有授权媒体可通过 `organize` 进入统一流程。

## 6. TAC-mini 与 TAC++

正式配置是 `configs/data/tac_mini.yaml`。renderer 当前支持：

- 按模板采样可重叠 sources；
- 100 ms 对齐 placement；
- gain、RMS activity、merge gap；
- 可选 source-specific RIR manifest；
- 保存 mixture、stems、ledger、caption、seed 和变换参数；
- 固定 seed 的音频字节级重放。

使用真实 RIR 时，把它们整理成普通 source manifest，并在配置中增加：

```yaml
rir_manifest: /data/sceneledger/manifests/rirs.jsonl
rir_probability: 0.7
```

完整 TAC++ 还需后续加入 device response、codec、ducking、occlusion 和从 web 统计拟合的 scene prior；这些不应在第一轮 500 条 smoke data 前阻塞基础复现。

## 7. Exact-CARC

```bash
sceneledger carc \
  --backgrounds /data/sceneledger/manifests/web_backgrounds.jsonl \
  --sources /data/sceneledger/manifests/known_sources.jsonl \
  --config configs/data/exact_carc.yaml \
  --output /data/sceneledger/derived/exact_carc_v1
```

每行 `pairs.jsonl` 保存 `before`、`after`、精确 target stem、placement sample、SNR、master scale、delta event 和 audibility target。`after = before + target`，因此同一对可同时用于 add 和 exact removal，不需要 separator。

## 8. MOSS-Audio 基线

初始化服务器：

```bash
SCENELEDGER_PROJECT_ROOT=$(pwd) bash scripts/setup_server.sh
conda activate sceneledger
bash scripts/download_models.sh
```

官方 MOSS fine-tuning 数据格式由 `sceneledger moss-sft` 生成。启动 LoRA：

```bash
export MOSS_ROOT=$(pwd)/third_party/MOSS-Audio
export MOSS_MODEL_DIR=$(pwd)/weights/MOSS-Audio-4B-Instruct
export TRAIN_JSONL=/data/sceneledger/derived/tac_v1/moss_train.jsonl
export OUTPUT_DIR=$(pwd)/outputs/b2_moss_tac
bash scripts/run_moss_lora.sh
```

V100 等不支持 FlashAttention 的 GPU 设置：

```bash
export ATTN_IMPLEMENTATION=eager
```

单条 B0 推理：

```bash
sceneledger moss-infer \
  --upstream-root third_party/MOSS-Audio \
  --model-path weights/MOSS-Audio-4B-Instruct \
  --audio example.wav \
  --prompt "Describe all audible speech, lyrics, music, and sounds with timestamps."
```

## 9. Track–Event slot 结构训练

先用官方 MOSS encoder 提取 12.5 Hz 特征，再插值到 10 Hz ledger 网格：

```bash
python scripts/extract_moss_features.py \
  --upstream-root third_party/MOSS-Audio \
  --model-path weights/MOSS-Audio-4B-Instruct \
  --ledgers /data/sceneledger/derived/tac_v1/ledgers.jsonl \
  --render-manifest /data/sceneledger/derived/tac_v1/render_manifest.jsonl \
  --output /data/sceneledger/features/tac_v1

python scripts/train_slots.py \
  --features /data/sceneledger/features/tac_v1 \
  --config configs/model/track_event_slots.yaml \
  --output outputs/s1_slots
```

NPZ 保存 `features`、track/event type、activity 和 event-to-track target。训练使用两级 Hungarian matching、activity BCE+Dice、presence/eventness、pointer 和 containment loss。

## 10. SAM Audio 教师

SAM Audio 是可选组件。按官方仓库申请 checkpoint 权限并安装后，使用：

```python
from sceneledger.integrations.sam_audio import SamAudioAdapter

separator = SamAudioAdapter("facebook/sam-audio-large")
target, residual, sample_rate = separator.separate(
    "mixture.wav", "glass breaking", predict_spans=True, reranking_candidates=8
)
```

separator 输出不得直接作为标签，仍需 target/residual margin、重构、重复轨和人工 precision audit。

## 11. 评价

```bash
sceneledger validate predictions.jsonl
sceneledger evaluate references.jsonl predictions.jsonl --output reports/metrics.json
```

当前实现报告 event precision/recall/F1、semantic token F1、multi-span tIoU、onset/offset MAE、track pointer accuracy、source-count MAE 和 per-type 指标。ASR WER、歌词指标与 hallucination counterfactual suite 应在接入实际模型后追加。
