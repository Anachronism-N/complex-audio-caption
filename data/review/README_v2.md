# V2 音频质量 Review 指南

## 目标
确认改进后的 v2 mixture 音频听上去是否更自然、更复杂、无明显 bug。

## 改进点（vs v1）
1. **背景铺底**：无环境音源的 clip 自动添加 -18dB 环境音 → 消除静音段
2. **复杂模板**：新增 complex_cocktail（5-6源）、rich_band（5源）、multi_event_dense（4-5源）
3. **ducking 随机化**：2-5dB 深度，30% 概率不 ducking
4. **按类型 fade**：环境音 1.5s、音乐 0.5s、音效 50ms
5. **改进 ambience**：粉红噪声 + 低通 + 调制 + 滴答声
6. **按模板设时长**：isolated_sfx 3-8s、repeated_event 5-10s

## Review 文件
- **CSV**：`data/review/review_v2_audio_quality.csv`（20 条，覆盖全部 13 个模板）
- **音频**：`/tmp/b3_v2/audio/mix_XXXXXX.wav`

## Review 步骤
1. 打开 CSV（Excel/Sheets）
2. 逐条听音频 `audio/mix_XXXXXX.wav`
3. 填写以下列：

| 列 | 说明 | 取值 |
|---|---|---|
| `audio_natural` | 整体听感是否自然 | Y/N/partial |
| `silence_issue` | 是否有明显静音段 | Y/N |
| `boundary_issue` | 音轨边界是否突然消失 | Y/N |
| `mixing_issue` | 混音是否有问题（盖住、爆音等） | Y/N |
| `notes` | 具体发现 | 文字 |

## 重点关注
1. **背景铺底是否有效**：`has_background_fill=Y` 的 clip 是否还有长静音？
2. **复杂模板是否合理**：complex_cocktail/rich_band/multi_event_dense 是否听起来像真实复杂场景？
3. **ducking 是否自然**：有语音的 clip（`has_ducking=Y`）中，语音是否清晰？音乐是否被过度压低？
4. **边界是否平滑**：环境音/音乐的 fade 是否自然（vs v1 的突然消失）？
5. **ambience 改进**：环境音是否比 v1 更像雨声/环境声？

## CSV 列说明
| 列 | 说明 |
|---|---|
| clip_id | 音频文件名 |
| template | 场景模板 |
| audio_path | 音频相对路径 |
| duration | 时长（秒） |
| n_sources | 源数 |
| n_events | 事件数 |
| sources | 源列表（类型(ID) onset=时间 gain=增益） |
| events | 事件列表（类型 [时间] 文本） |
| has_background_fill | 是否有自动背景铺底 |
| has_ducking | 是否有 ducking（有语音/人声时） |
| audio_natural | **人工填写**：整体听感是否自然 |
| silence_issue | **人工填写**：是否有静音问题 |
| boundary_issue | **人工填写**：是否有边界突然消失 |
| mixing_issue | **人工填写**：是否有混音问题 |
| notes | **人工填写**：具体发现 |
