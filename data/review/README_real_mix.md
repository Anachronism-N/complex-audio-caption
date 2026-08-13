# 真实音频混合数据 Review 指南

## 数据位置

| 文件 | 说明 |
|---|---|
| `/tmp/real_mix/audio/realmix_XXXX.wav` | 200 条真实音频混合（8s each） |
| `data/derived/real_mix/manifest.jsonl` | manifest（含 MOSS 逐源 caption） |
| `data/derived/real_mix/review.csv` | review 表（Excel/Sheets 打开） |

## 数据构成

每条 mixture 由 2-3 个 ESC-50 真实音效混合而成：
- **音源**：ESC-50 数据集（2000 条 CC 授权真实环境音，50 类）
- **caption**：MOSS-Audio 零样本逐源生成（非固定短语）
- **混合**：随机 gain/onset/fade/RIR
- **时长**：8s 固定

## Review 步骤

### 1. 打开 review CSV
用 Excel/Google Sheets 打开 `data/derived/real_mix/review.csv`

### 2. 逐条 review
对每条 clip：

1. **听混合音频**：打开 `/tmp/real_mix/audio/realmix_XXXX.wav`
2. **检查音源 caption**：`source_captions` 列是 MOSS 对每个源的描述
   - 确认：你听到的声音与 caption 描述是否匹配？
   - 例如：caption 说"dog barking"，你是否听到了狗叫？
3. **检查混合质量**：音轨混合后是否自然？
4. **填写**：
   - `audio_natural`：混合后听感是否自然（Y/N/partial）
   - `caption_accurate`：MOSS 的描述是否准确（Y/N/partial）
   - `notes`：具体发现

### 3. 重点关注

| 关注点 | 说明 |
|---|---|
| **caption 准确性** | MOSS 零样本描述是否与实际听到的声音匹配？ |
| **混合自然度** | 2-3 个真实音效混合后是否像真实场景？ |
| **音色识别** | 能否辨认出具体声音（狗叫、雨声、敲门等）？ |
| **时间对齐** | caption 中的时间是否与实际听到的事件时间匹配？ |
| **混合问题** | 是否有爆音、源互相盖住、不自然过渡等？ |

### 4. 与合成数据对比

这是**第一次用真实音频**做 review。与之前合成数据的关键区别：
- ✅ 音色可辨认（真实狗叫 vs 合成噪声burst）
- ✅ caption 丰富（MOSS 生成 vs 固定短语）
- ✅ 多样性高（50 类不同声音 vs 4 种波形）

## CSV 列说明

| 列 | 说明 |
|---|---|
| clip_id | mixture ID |
| audio_path | 音频路径 |
| duration | 时长（8s） |
| n_sources | 源数（2-3） |
| source_categories | 源类别（如 dog \| rain） |
| source_captions | MOSS 对每个源的描述 |
| audio_natural | **人工填写**：混合是否自然 |
| caption_accurate | **人工填写**：caption 是否准确 |
| notes | **人工填写**：具体发现 |

## Review 完成后

告诉我你的发现，我会：
1. 统计 MOSS caption 准确率
2. 分析混合质量问题
3. 根据反馈调整混合参数
4. 用这些数据训练/评估模型
