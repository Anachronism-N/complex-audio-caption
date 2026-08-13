# 真实音频混合 v2 Review 反馈（2026-08-13）

## Review 结果

人工 review 10 条含音乐/人声的 v2 混合音频后发现：

### 问题 13：人声不够清晰
- **现状**：部分 clip 能听到人声，但内容不清晰
- **原因**：
  1. 人声源（test_en.mp3, faker_and_chovy.mp3）本身可能有背景噪声
  2. 混音时人声 gain 与 sfx/music gain 未合理平衡
  3. 缺少人声增强（如 compression、EQ 提升 1-3kHz）
  4. ducking 可能不够（-4dB 可能不足以让人声突出）
- **改进**：
  - 人声 gain 提高 3-6dB（相对于其他源）
  - 加入人声压缩（dynamic range compression）
  - EQ 提升 1-3kHz（人声清晰度频段）
  - ducking 深度增加到 6-8dB
  - 人声源改用更干净的录音（如 LibriSpeech）

### 问题 14：音源库过于单一
- **现状**：
  - SFX：ESC-50（50 类，每类 40 条）—— 类别有限
  - 音乐：仅 2 条 MOSS demo（qilixiang, game）—— 风格单一
  - 人声：仅 2 条 MOSS demo（test_en, faker_and_chovy）—— 说话人单一
- **问题**：
  - 200 条混合中有大量重复音源
  - 音乐只有 2 首歌 → 多样性不足
  - 人声只有 2 个说话人 → 无法学习多说话人区分
  - ESC-50 每类只有 40 条 → 容易过拟合
- **需要**：
  - 更多音乐：不同流派（古典、摇滚、电子、爵士、民谣）
  - 更多人声：不同性别、年龄、语言、语速
  - 更多音效：Freesound CC0、UrbanSound8K、FSD50K
  - 目标：每类 ≥100 条不同音源

## 改进方案

### 方案 F：扩展音源库

| 音源类型 | 当前 | 目标 | 数据集 |
|---|---|---|---|
| SFX | ESC-50 (2000) | ESC-50 + Freesound CC0 (5000+) | Freesound API |
| 音乐 | 2 条 demo | 100+ 首不同流派 | FMA dataset / MUSDB18 |
| 人声 | 2 条 demo | 100+ 说话人 | LibriSpeech / Common Voice |
| 环境音 | ESC-50 ambience | + UrbanSound8K | UrbanSound8K |

### 方案 G：改进混音参数

1. **人声增强**：
   - gain +3-6dB（相对于其他源）
   - 压缩：threshold=-20dB, ratio=3:1
   - EQ：+2dB @ 2kHz（清晰度）

2. **ducking 加强**：
   - 深度 6-8dB（当前 4dB 不够）
   - 快速 attack/release

3. **音乐处理**：
   - 根据场景调整音乐 gain（咖啡店 -10dB，演唱会 0dB）
   - 不同流派不同 EQ

4. **音效处理**：
   - 根据 distance 调整 gain + reverb
   - 远处音效：更低 gain + 更多 reverb

### 方案 H：LLM 辅助场景+混音参数设计

用 LLM 一次性生成完整场景描述+混音参数：

```python
prompt = f"""
Design a realistic audio scene with 3-4 sources.
For each source, specify:
- type (speech/music/sfx/ambience)
- description
- gain_db (relative)
- onset_s, duration_s
- ducking (if speech, how much to lower others)
- reverb_t60

Scene requirements:
- Logical real-world scenario
- Varied source types
- Natural mixing levels
"""
```

## 下一步行动

1. **下载更多音源**：
   - LibriSpeech（人声+转录，1000+ 说话人）
   - FMA subset（音乐，多种流派）
   - UrbanSound8K（城市音效）

2. **改进混音参数**：
   - 人声增强（gain+compression+EQ）
   - ducking 加深到 6-8dB
   - 按场景调整各源 gain

3. **LLM 辅助场景生成**：
   - 生成更丰富的场景模板
   - 自动指定混音参数

## 对论文的影响

当前音源库虽然比合成数据好很多，但仍不足以训练一个鲁棒的复杂音频 caption 模型：
- **训练数据多样性不足** → 模型可能过拟合到特定音源
- **人声清晰度不够** → speech SA-WER 指标会受影响
- **需要更大规模真实音源库** → 这是 WildMix-Cap benchmark 的前置条件
