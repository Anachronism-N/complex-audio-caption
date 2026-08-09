# 复现指南

> 2026-08-09 更新：仓库中旧的 500 条 TAC-mini manifest 是 renderer v0.2
> 历史产物，只用于追溯 B0，不可继续训练。请按本文重新渲染 v0.3；详细状态见
> `docs/13_data_pipeline_status.md`。

> 本文档说明如何从空环境复现 TAC-mini 数据集和 B0 基线。所有步骤均可确定性重放。

## 1. 环境准备

### 1.1 基础环境（协议、解析、指标、数据渲染）

```bash
pip install -e .
# 依赖：pydantic>=2.6, numpy>=1.24, scipy>=1.10, PyYAML>=6.0, soundfile, librosa
# 测试：pytest
```

### 1.2 MOSS-Audio 环境（B0 推理，可选）

MOSS-Audio-4B 需要独立的 conda 环境（pin `transformers==4.57.1` / `numpy>=2.0` / `torch==2.9.1`）：

```bash
# 克隆 MOSS-Audio 仓库（不安装，只读引用）
git clone https://github.com/OpenMOSS/MOSS-Audio.git third_party/MOSS-Audio
git -C third_party/MOSS-Audio checkout 5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883

# 建独立环境
conda create -n moss-audio python=3.12 -y
conda run -n moss-audio pip install torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n moss-audio pip install \
    "transformers==4.57.1" "numpy>=2.0" safetensors soundfile tiktoken \
    einops scipy tqdm packaging accelerate peft librosa "pydantic>=2.6"
conda run -n moss-audio pip install --no-deps -e third_party/MOSS-Audio
conda run -n moss-audio pip install --no-deps -e .

# 下载权重（~10.5 GB）
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('OpenMOSS-Team/MOSS-Audio-4B-Instruct', local_dir='/tmp/moss_weights')"
```

**已知坑**（均已在 `moss_adapter.py` 中处理）：
- `torchaudio.load` 在 2.9.x 需要 `torchcodec`——adapter 改用 soundfile + `scipy.signal.resample_poly`
- `MossAudioModel` 类不在 HF `auto_map`——adapter 把 `third_party/MOSS-Audio` 加入 `sys.path`
- 输入采样率 `mel_sr=16000`——adapter 自动从 24 kHz 重采样

## 2. 数据流程复现

### 2.1 渲染 TAC-mini（500 条，确定性）

```bash
# 渲染到本地磁盘（cephfs 写小文件极慢，用 /tmp）
python -m sceneledger.cli.render \
    --config configs/data/tac_mini.yaml \
    --output-dir /tmp/tac_mini \
    --validate
```

**验收**（`docs/11` P2 gate）：
- `replay ok=500/500`——同 seed 重渲染 hash 一致
- `stems_sum ok=500`——stem 求和等于 dry mixture
- `ledger_valid=500`——所有目标 ledger 通过 schema 校验
- 产物：`/tmp/tac_mini/manifest.jsonl` + `data_card.md` + `listen_list.csv` + `audio/*.wav`

### 2.2 数据集说明

- **合成源池**（`SyntheticSourcePool`）：用正弦/噪声/和弦生成占位音频，无需许可语料即可完整复现。caption 文本为占位中文/英文短语。
- **文件源池**（`FileSourcePool`）：在 `configs/data/tac_mini.yaml` 中设 `pool.kind: file` 并指向 LibriSpeech/Freesound/MUSDB18 即可渲染真实 R1 数据集。
- **确定性**：每个 scene 由 `seed` 完全决定；同一 seed + 同一 pool 产生 bit-identical 的 waveform。
- **无泄漏 split**：`datamodule.group_split` 按 source path 分组，同源不跨 fold。

## 3. B0 基线复现

### 3.1 真实 MOSS B0

```bash
conda run -n moss-audio python -m sceneledger.cli.infer \
    --manifest data/derived/tac_mini/manifest.jsonl \
    --audio-base /tmp/tac_mini \
    --backend moss \
    --model-path /tmp/moss_weights \
    --device cuda:0 --dtype bfloat16 \
    --output reports/b0_predictions_moss.jsonl \
    --report reports/b0_infer_report_moss.json
```

### 3.2 评估

```bash
python -m sceneledger.cli.evaluate \
    --prediction reports/b0_predictions_moss.jsonl \
    --reference data/derived/tac_mini/manifest.jsonl \
    --output reports/b0_metrics_moss.json --pretty
```

### 3.3 鲁棒性分层

```bash
python -c "from sceneledger.eval.robustness import robustness_report; \
    robustness_report('reports/b0_metrics_moss.json', \
    'data/derived/tac_mini/manifest.jsonl', \
    'reports/b0_robustness.json')"
```

## 4. B0 预期结果

| 指标 | 预期值 | 说明 |
|---|---|---|
| strict-format-success | 0.0% | 零样本 MOSS 不输出 XML/atomic-token 格式 |
| event-F1 | 0.000 | 解析器无法从自由文本恢复规范事件 |
| 定性输出 | 多样 | 模型能描述音频并带时间戳，但格式不统一（`[0.00]`、`[0.000-1.000 sec]`、自然语言） |

B0 的 F1=0 是**预期且正确的基线**：它证明模型具备描述能力但缺乏格式纪律，B1 SFT 的目标就是教会规范格式。

## 5. Mock B0（无需模型，验证流水线）

```bash
python -m sceneledger.cli.infer \
    --manifest data/derived/tac_mini/manifest.jsonl \
    --audio-base /tmp/tac_mini \
    --backend mock \
    --output reports/b0_predictions.jsonl \
    --report reports/b0_infer_report.json
```

Mock 适配器从目标 ledger 生成确定性扰动预测（遗漏/幻觉/边界偏移），用于在无 GPU 时验证 infer→parse→evaluate 全链路。预期 event-F1≈0.88。
