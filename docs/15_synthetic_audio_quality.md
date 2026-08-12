# 合成音频质量问题与改进方案

> 基于 mix_000194 等 clip 的人工 review 发现（2026-08-12）

## 背景

人工 review 的目标是**确认 mixture 音频听上去是否正确自然**，不是检查模型预测（预测错误可通过指标自动检测）。Review 发现以下 5 个问题。

## 问题诊断

### 问题 1：音色不像真实声音
- **现状**：`SyntheticSourcePool` 用纯数学函数生成音频（正弦波=音乐、白噪声=雨声、噪声burst=音效）
- **听感**：像电子蜂鸣/嘈杂嗡嗡声，不像真实乐器/雨声/人声
- **案例**：mix_000194 的 GT 标注为"雨声背景"，但听上去是一片嘈杂的声音，无法辨认是雨声
- **根因**：无真实音频采样，纯数学合成无法还原真实音色

### 问题 2：音轨边界突然消失
- **现状**：fade in/out 仅 10ms
- **听感**：音轨突然出现/消失，不自然
- **真实情况**：雨声渐弱、音乐渐入、语音有呼吸间隔
- **案例**：ambience 在 offset 处突然消失，真实环境中环境音会逐渐衰减

### 问题 3：环境音不持续 + 静音段过多
- **现状**：ambience 在 onset 处突然开始，在 offset 处突然结束；长 clip 中大量静音
- **听感**：0-12.4s 有"雨声"，12.4s 后突然静音
- **真实情况**：环境音通常是整段持续，有自然起伏；不应有长静音段
- **案例**：25s 的 clip 只有 1 个 0.3s 的音效，中间全是静音

### 问题 4：时长与事件数不匹配
- **现状**：所有模板共用 10-30s 时长范围
- **听感**：isolated_sfx 可能 25s 只有 1 个 0.3s 事件 → 大量静音，不像"复杂 caption"
- **根因**：`duration_range = (10, 30)` 对所有模板统一，未按模板调整

### 问题 5：混音参数过于简单
- **现状**：只有 gain_db + onset + RIR + echo + 10ms fade
- **缺失**：
  - ducking（语音时降低音乐增益 3-6dB）
  - 侧链压缩
  - 自然包络（渐入渐出时间应按源类型不同）
  - 低频/高频架
  - 环境音应铺底持续，而非突然消失
- **改进方向**：除规则外，可考虑用 LLM 根据场景描述生成混音参数

## 改进方案

### 方案 A：改进合成器（快速，不依赖外部数据）

1. **按模板设置时长范围**：
   ```python
   template_duration = {
       "isolated_sfx": (3, 8),
       "repeated_event": (5, 10),
       "ambient_with_intermittent_sfx": (10, 20),
       "speech_over_music": (10, 25),
       "music_with_sfx": (10, 25),
       "speech_music_sfx": (10, 30),
       "lyrics_over_music": (10, 25),
       "speech_music_lyrics_sfx": (15, 30),
       "overlapping_speakers": (5, 15),
       "random_mix": (10, 25),
   }
   ```

2. **加长 fade**：环境音 1-2s 渐入渐出，音效 50-100ms，音乐 0.5-1s

3. **改进 ambience 合成**：
   - 雨声：粉红噪声 + 周期性"滴答"调制 + 低通
   - 风声：棕色噪声 + 缓慢振幅调制
   - 室内环境：粉红噪声 + 极低增益 + 偶尔吱嘎声

4. **改进音效合成**：
   - 玻璃破碎：多个高频噪声 burst + 快速衰减 + 混响尾
   - 撞击：低频 thump + 高频 crack + 衰减
   - 门响：吱嘎声（频率扫描）+ 关门 thud

5. **背景填充**：长 clip 中添加低增益环境音铺底，避免静音段

6. **ducking**：语音 onset 时自动降低音乐增益 3-6dB

### 方案 B：接入真实音效库（中等，需许可语料）

用 Freesound CC0 音频替换合成音频：
1. 下载 Freesound CC0 分类音效（雨声、玻璃、犬吠等）
2. `FileSourcePool` 直接读取真实音频文件
3. renderer 的混音逻辑不变，只替换源音频

**优势**：音色真实，无需改进合成器
**劣势**：需下载和管理许可音频

### 方案 C：LLM 辅助混音参数设计（高级）

用 LLM 根据场景描述生成混音参数：
```python
# LLM 输入：场景描述 + 源类型列表
# LLM 输出：每个源的 gain/fade/ducking/reverb 参数
prompt = f"""
Scene: {template}, sources: {source_kinds}, duration: {duration}s
Suggest mixing parameters for each source:
- gain_db: relative level
- fade_in_s, fade_out_s: onset/offset envelope
- ducking: when to lower this source for others
- reverb_t60: room characteristics
"""
```

**优势**：混音更自然、更多样化
**劣势**：增加 LLM 调用成本和复杂度

## 推荐路径

1. **立即做**：方案 A（改进合成器）——解决时长、fade、静音问题（已部分实现：按模板设置时长范围）
2. **短期**：方案 B（接入 Freesound CC0）——解决音色问题
3. **中期**：方案 C（LLM 辅助混音）——解决混音自然度

## 对论文的影响

合成音频的音色不像真实声音，这意味着：
- **模型学到的是时间结构**（事件何时出现/消失），不是音色识别
- **在真实音频上泛化能力存疑**——需要真实语料验证
- **论文应诚实说明**：合成数据用于验证格式和结构，真实数据验证泛化

## Review 目标澄清

人工 review 的目标是**确认 mixture 音频听上去是否正确自然**：
- ✅ 检查：混音后音频是否听起来合理（音色、边界、静音段）
- ✅ 检查：渲染是否有 bug（爆音、完全静音、重复噪声）
- ❌ 不需要：检查模型预测是否正确（可由指标自动检测）
- ❌ 不需要：精确判断时间精度（可由 onset-MAE 等指标量化）
- ❌ 不需要：检查 GT 是否正确（GT 由渲染过程确定性生成，理论上无错）

## 已实施的改进

1. **按模板设置时长范围**（`scene_graph_sampler.py`）：
   - isolated_sfx: 3-8s（避免长静音）
   - repeated_event: 5-10s
   - overlapping_speakers: 5-15s
   - 其他模板保持 10-30s
