# 真实音频混合 v2 Review 指南（含音乐/人声）

## 改进点（vs v1）

| 改进 | v1 | v2 |
|---|---|---|
| 源选择 | 随机 2-3 个 ESC-50 | 12 个预定义真实场景模板 |
| 音乐 | 无 | MOSS demo 音频（qilixiang.mp3, game.mp3） |
| 人声 | 无 | MOSS demo 音频（test_en.mp3, faker_and_chovy.mp3） |
| 场景逻辑 | 无（如 clock_alarm+laughing+airplane） | 有（如 coffee_shop=音乐+咖啡机+顾客） |
| ducking | 无 | 有（语音时音乐降 4dB，70% 概率） |

## Review 文件

| 文件 | 说明 |
|---|---|
| `data/derived/real_mix_v2/review_10.csv` | 10 条含音乐/人声的 clip |
| `data/derived/real_mix_v2/audio/rv2_XXXX.wav` | 音频文件 |

## 10 条 Review Clip

| clip_id | 场景 | 源组成 |
|---|---|---|
| rv2_0043 | speech_with_sfx | 人声 + 音效 |
| rv2_0009 | coffee_shop | 音乐 + 咖啡机 + 顾客 |
| rv2_0097 | coffee_shop | 音乐 + 咖啡机 + 顾客 |
| rv2_0088 | speech_with_sfx | 人声 + 音效 |
| rv2_0069 | speech_with_sfx | 人声 + 音效 |
| rv2_0021 | speech_with_music | 人声 + 背景音乐（含 ducking） |
| rv2_0124 | speech_with_sfx | 人声 + 音效 |
| rv2_0015 | speech_with_sfx | 人声 + 音效 |
| rv2_0115 | coffee_shop | 音乐 + 咖啡机 + 顾客 |
| rv2_0179 | speech_with_sfx | 人声 + 音效 |

## Review 步骤

1. 打开 `data/derived/real_mix_v2/review_10.csv`（Excel/Sheets）
2. 逐条听 `data/derived/real_mix_v2/audio/rv2_XXXX.wav`
3. 检查：
   - `source_X_caption` 是否与实际听到的声音匹配？
   - 音乐/人声是否清晰可辨？
   - ducking 是否有效（rv2_0021 人声+音乐）？
   - 混合是否自然像真实场景？
4. 填写：
   - `audio_natural`：混合是否自然（Y/N/partial）
   - `caption_accurate`：描述是否准确（Y/N/partial）
   - `notes`：具体发现

## 重点关注

1. **音乐可辨认吗？**（qilixiang/game 音乐片段）
2. **人声可辨认吗？**（English/Korean 语音）
3. **ducking 效果**（rv2_0021：语音时音乐是否降低？）
4. **场景合理性**（coffee_shop 听起来像咖啡店吗？）
