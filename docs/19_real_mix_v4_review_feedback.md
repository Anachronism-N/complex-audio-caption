# 真实音频混合 v4 Review 反馈（2026-08-13）

## Review 结果

人工 review v4 发现严重问题：**音频内容与 caption 描述完全不匹配**。

### 问题 18：音频内容与 caption 描述不匹配（CRITICAL）
- **rv4_0054**（speech_with_music）：
  - caption 说：人声 + 音乐
  - 实际听到：**前后两段相同的人声，没有音乐**
- **rv4_0048**（concert_outdoor）：
  - caption 说：音乐 + 掌声 + 风
  - 实际听到：**笑声 + 人声 + 风声**（没有音乐）

**根因分析**：
1. **源加载错误**：scene 模板指定了 music/sfx 类型，但实际加载的音频文件可能不匹配
2. **MOSS caption 描述的是原始音源，但混音后听到的是不同内容**：
   - MOSS 对每个 ESC-50 源生成 caption（如"笑声"）
   - 但 manifest 里的 caption 用的是 scene 模板的描述（如"音乐"）
   - 导致 caption 与实际音频内容完全脱节
3. **音乐源可能加载失败**：game.mp3 可能没有成功混入

### 问题 19：人声音量仍然偏小
- **rv4_0048**：人声音量较小，即使 v4 已经 speech gain +3dB
- **根因**：
  - 人声源（test_en.mp3）本身录音音量低
  - 或者 ducking 逻辑可能反而降低了人声
  - 压缩后峰值降低，感知音量下降

### 问题 20：caption 来源不一致
- **当前逻辑**：
  - scene 模板预定义描述（如"background music"）→ 写入 manifest
  - MOSS 对 ESC-50 源生成真实 caption → 也写入 manifest
  - 但两者可能不一致
- **应该**：caption 必须描述实际听到的内容，不是模板预设

## 根因总结

v4 的核心问题是 **caption 与音频内容脱节**：
```
scene 模板预设 "music + applause + wind"
→ 但实际加载的源可能是 "laughter + speech + wind"
→ caption 说"音乐"但听到的是"笑声"
```

这说明：
1. scene 模板的 `desc` 字段被当作 caption 使用（错误）
2. 实际加载的音源与模板描述不匹配
3. MOSS 对 ESC-50 源生成的 caption 才是准确的，但没有被用作最终 caption

## 改进方案

### 方案 M：统一 caption 来源
1. **混音后**对整个 mixture 运行 MOSS → 生成 mixture caption
2. **每个源**的 caption 来自 MOSS 对该源的实际推理（不是模板预设）
3. **manifest 中的 caption = MOSS 对实际音源的描述**

### 方案 N：验证音频与 caption 匹配
1. 混音后检查每个源的 RMS → 确认源实际存在
2. 如果某源 RMS 过低 → 标记为"未混入"，不写入 caption
3. 添加音频指纹验证

### 方案 O：修正源加载逻辑
1. 检查 music 源是否成功加载（game.mp3 路径是否正确）
2. 添加日志：每个源的文件路径、时长、RMS
3. 混音后验证各源贡献

## 下一步行动

1. **修正 caption 来源**：用 MOSS 对实际音源的描述作为 caption
2. **添加混音后验证**：检查每个源的 RMS
3. **修正源加载**：确保 music 源正确加载
4. **人声增益**：进一步提高人声 gain 或使用更干净的人声源
