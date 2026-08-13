# 真实音频混合 v3b Review 反馈（2026-08-13）

## Review 结果

人工 review 10 条 v3b 含音乐/人声的混合音频后发现：

### 问题 15：音乐源混叠——两首音乐同时播放
- **rv3b_0021**：应该是 speech + music，但听起来像**两首歌同时播放**（七里香 + 另一首）
- **根因**：MOSS demo 的 `qilixiang.mp3` 本身是带人声的歌曲，不是纯音乐
  - 当它作为 "music" 源时，其人声与 "speech" 源的人声混淆
  - 听感上像两首歌同时播放
- **改进**：
  - 区分 "纯音乐" 和 "带人声音乐"
  - 使用 MUSDB18 分轨（纯伴奏）或纯音乐文件
  - 或者将带人声的音乐标记为 "vocal" 类型，不作为 "music"

### 问题 16：部分 clip 听不到音乐/人声
- **rv3b_0097**（coffee_shop）：标注有 music，但听不到音乐
- **根因**：
  - 音乐 gain 过低（-12dB in coffee_shop）被其他源盖住
  - 或者音乐源文件加载失败，实际没有混入
- **改进**：
  - 提高音乐 gain（-12dB → -6dB）
  - 验证音乐源是否成功加载
  - 添加混音后的 RMS 检查

### 问题 17：音效过大覆盖人声
- **rv3b_0043**：前面有人声，但后续音效过大几乎覆盖人声
- **根因**：
  - sfx gain（-3dB）与人声 gain（0dB）差距不够
  - ducking 只在 speech+music 时触发，speech+sfx 时不 ducking
  - 人声压缩后峰值降低，相对 sfx 更弱
- **改进**：
  - speech gain 提高到 +3-6dB（vs sfx -3dB → 差距 6-9dB）
  - ducking 扩展到 speech+sfx 场景（不仅 speech+music）
  - 或者 sfx gain 降低到 -9dB

### 正面发现
- **rv3b_0015**：确实有人声，人声可辨认
- 说明人声源本身是可用的，问题在于混音参数

## 改进方案

### 方案 I：修正音乐源
1. 下载纯音乐（无人声）——MUSDB18 的 accompaniment 分轨
2. 或将 qilixiang.mp3 标记为 "vocal"（带人声音乐）
3. game.mp3 保留为纯音乐

### 方案 J：调整混音参数
| 源类型 | 旧 gain | 新 gain | 说明 |
|---|---|---|---|
| speech | 0dB | **+3dB** | 提高人声 |
| music | -12dB（咖啡店） | **-6dB** | 提高音乐可听度 |
| sfx | -3dB | **-6dB** | 降低音效避免覆盖人声 |
| ambience | -15dB | -15dB | 保持低 |

### 方案 K：扩展 ducking
- 旧：仅 speech+music 时 ducking
- 新：**speech 存在时** ducking 所有其他源（music + sfx + ambience）
- 深度：6-8dB

### 方案 L：混音后验证
- 混音后计算各源的 RMS 比值
- 确保 speech RMS > sfx RMS + 3dB
- 如果不满足，自动调整 gain

## 下一步行动

1. **修正音乐源**：区分纯音乐和带人声音乐
2. **调整 gain**：speech +3dB, music -6dB, sfx -6dB
3. **扩展 ducking**：speech 时 ducking 所有其他源
4. **混音后验证**：RMS 检查确保人声可听
