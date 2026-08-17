# LLM 辅助混音方案 + 未解决问题分析

> 文档日期：2026-08-17

## 1. 为什么没有使用 LLM 辅助混音？

### 1.1 当前状态

当前混音管道完全基于**规则**：
- 12 个预定义场景模板（硬编码源类型组合）
- 固定 gain 范围（speech +0~6dB, music -6~0dB, sfx -9~-3dB）
- 固定 ducking 深度（6-8dB）
- 固定 fade 时间（ambience 1.5s, music 0.5s, sfx 50ms）
- ESC-50 类别按场景硬编码限制（如餐厅只能用 can_opening/laughing/coughing）

### 1.2 为什么没做 LLM 辅助？

**实际原因**：
1. **优先级**：先验证"真实音频能否训练"这个核心问题，再优化混音质量
2. **时间**：LLM 辅助需要设计 prompt、调用 API、解析输出，工程量较大
3. **不确定性**：不确定 LLM 生成的混音参数是否比规则更好
4. **MOSS 已经在做 caption**：MOSS 对每个源生成 caption 已经是 LLM 辅助的一部分，但混音参数本身没用 LLM

**这是正确的决策吗？**
- ✅ 正确：先验证管道可行性（v6k F1=0.970 证明了管道有效）
- ❌ 不足：混音质量问题（docs/22 的问题 25-29）可能需要 LLM 来解决

## 2. 之前存在的问题哪些没解决？

### 2.1 未解决问题清单

| # | 问题 | 状态 | 未解决原因 |
|---|---|---|---|
| 25 | 时间戳起始偏后 | ⚠️ 部分解决 | v7/v8 用 RMS 活跃区间修复了 onset，但 F1 下降。根因是 RMS 检测不够精确（阈值 0.005 太低/太高），需要更精细的活动检测 |
| 26 | caption 不够详细 | ⚠️ 部分解决 | v7 用增强 prompt 修复，但导致幻觉增加。根因是 MOSS 对同一音源可能生成不同描述，需要一致性控制 |
| 27 | 音源内容与 caption 不匹配 | ⚠️ 部分解决 | v5 用 MOSS 逐源 caption 修复，但 ESC-50 文件标签本身可能不准确。需要人工验证或 LLM 验证 |
| 28 | 人声质量需提升 | ❌ 未解决 | LibriSpeech 下载失败（HF 仓库结构问题），MOSS demo 音频本身带噪声且只有 2 个说话人 |
| 29 | 场景组合不合理 | ⚠️ 部分解决 | v7 限制了场景的 ESC-50 类别，但场景模板本身是硬编码的，不够灵活 |
| 9 | 合成音色不可辨认 | ✅ 已解决 | 接入真实音频（ESC-50 + GTZAN） |
| 15 | 音乐源含人声 | ✅ 已解决 | 替换为 GTZAN 纯器乐 |
| 18 | caption-audio 不匹配 | ✅ 已解决 | MOSS 逐源 caption |
| 21 | 同一人声重复 | ✅ 已解决 | 替换 game.mp3 |
| 22 | 音乐片段选取不佳 | ⚠️ 部分解决 | 随机选取仍可能切在不好的位置，需要智能选取 |

### 2.2 根本原因分析

**为什么这些问题没解决？**

1. **LibriSpeech 下载失败**：
   - HF 仓库 `openslr/librispeech_asr` 的 `allow_patterns=['dev-clean/*']` 没有匹配到文件
   - 实际文件在 `dev-clean/<speaker>/<chapter>/<file>.flac` 路径下
   - 需要修正 `allow_patterns` 或直接从 OpenSLR 下载 tar.gz

2. **RMS 活动检测不够精确**：
   - 当前用固定阈值 0.005 + 50ms 帧检测活动
   - 对于低音量源（如远处音效）可能检测不到
   - 对于高背景噪声源可能误检
   - 需要 adaptive threshold 或更精细的 VAD

3. **场景模板硬编码**：
   - 12 个模板是手动设计的，覆盖有限
   - 无法生成模板外的组合（如"海滩 + 海鸥 + 船笛"）
   - 需要 LLM 根据场景描述动态生成源组合

4. **人声多样性不足**：
   - 只有 2 个 MOSS demo 人声源（英文 + 韩文）
   - 没有不同性别、年龄、语速的说话人
   - 导致 speech F1 在 v8 下降（0.952→0.900）

5. **音乐片段随机选取**：
   - 从 GTZAN 随机选 8-12s 片段
   - 可能选到歌曲开头/结尾（不完整旋律）
   - 需要 LLM 或信号处理选择"有代表性的片段"

## 3. LLM 辅助混音方案

### 3.1 LLM 可以做什么？

| 环节 | 当前（规则） | LLM 辅助 | 预期改善 |
|---|---|---|---|
| 场景设计 | 12 个硬编码模板 | LLM 根据场景描述动态生成源组合 | 更多样的场景 |
| 源选择 | ESC-50 类别硬编码限制 | LLM 根据场景选择合理的音源类别 | 更合理的组合 |
| gain 设置 | 固定范围随机 | LLM 根据源类型+场景设置 gain | 更自然的混音 |
| ducking | 固定 6-8dB | LLM 根据语音/音乐关系设置 ducking 深度 | 更自然的 ducking |
| 音乐片段选取 | 随机 | LLM 选择"有代表性的片段" | 更好的音乐片段 |
| caption 生成 | MOSS 逐源 | LLM 根据混音参数+源信息生成 | 更一致的 caption |
| 时间戳 | RMS 检测 | LLM 根据源类型+活动检测设置 | 更准确的时间戳 |

### 3.2 具体方案

#### 方案 W：LLM 辅助场景+混音参数生成

```python
# 步骤 1：LLM 生成场景描述 + 源组合 + 混音参数
llm_prompt = f"""
Design a realistic audio scene for training audio captioning.

Output JSON with:
{{
  "scene_name": "string",
  "scene_description": "string",
  "duration_s": 8-12,
  "sources": [
    {{
      "role": "speech|music|sfx|ambience",
      "esc50_category": "string (if sfx/ambience)",
      "music_genre": "classical|jazz|blues (if music)",
      "gain_db": -15 to +6,
      "onset_s": 0 to duration-3,
      "duck_others": true|false,
      "fade_in_s": 0.01-2.0,
      "fade_out_s": 0.01-2.0
    }}
  ]
}}

Constraints:
- 2-4 sources per scene
- Sources should be logically related (e.g. coffee shop: music + coffee machine + customers)
- Speech gain should be higher than sfx gain
- Include 30% simple scenes (1-2 sources) and 70% complex (3-4 sources)
"""
```

#### 方案 X：LLM 辅助 caption 一致性

```python
# 步骤 2：LLM 检查 caption 与音频内容一致性
llm_check = f"""
Given:
- Scene: {scene_description}
- Source type: {source_type}
- MOSS caption: {moss_caption}
- Audio duration: {duration}s

Check:
1. Is the caption consistent with the scene? (e.g. "cricket" in "restaurant" = inconsistent)
2. Is the caption too long/short?
3. Does the caption describe the actual sound or just the label?

Output: {{
  "consistent": true|false,
  "corrected_caption": "string (if inconsistent)",
  "reason": "string"
}}
"""
```

#### 方案 Y：LLM 辅助时间戳

```python
# 步骤 3：LLM 根据源类型+活动检测设置时间戳
llm_timestamp = f"""
Given:
- Source type: {source_type} (speech/music/sfx/ambience)
- Audio RMS activity spans: {activity_spans}
- Placement onset: {onset}s

Determine:
- actual_onset: when the sound actually starts (considering fade-in)
- actual_offset: when the sound actually ends (considering fade-out)
- For continuous sounds (music/ambience): use full duration
- For transient sounds (sfx): use peak activity
- For speech: use voice activity

Output: {{"onset": float, "offset": float}}
"""
```

### 3.3 实施计划

| 步骤 | 内容 | 时间 |
|---|---|---|
| 1 | 修复 LibriSpeech 下载 | 30min |
| 2 | 实现 LLM 辅助场景生成（方案 W） | 2h |
| 3 | 实现 LLM caption 一致性检查（方案 X） | 1h |
| 4 | 实现 LLM 时间戳辅助（方案 Y） | 1h |
| 5 | 生成 1000 条 v9 数据 | 2h |
| 6 | 训练 + 评估 | 1h |
| 7 | 与 v6k 对比 | 30min |

### 3.4 预期效果

| 指标 | v6k（规则） | v9（LLM 辅助，预期） |
|---|---|---|
| F1 | 0.970 | 0.970+ |
| onset-MAE | 0.262s | < 0.200s |
| speech F1 | 0.952 | > 0.970 |
| 场景多样性 | 12 模板 | 无限 |
| caption 一致性 | 偶尔不匹配 | 检查后一致 |

## 4. 建议优先级

1. **修复 LibriSpeech 下载**（30min）→ 更多说话人
2. **LLM 辅助场景生成**（2h）→ 更多样的场景
3. **LLM caption 一致性检查**（1h）→ 更准确的 caption
4. **LLM 时间戳辅助**（1h）→ 更准确的 onset/offset
5. **生成 v9 + 训练**（3h）→ 验证效果
