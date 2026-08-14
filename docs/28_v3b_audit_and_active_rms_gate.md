# v3/v3b 审计与下一步 active-RMS 数据门禁

更新日期：2026-08-13

## 1. 结论

远端最新 v3/v3b 不能作为训练数据，也不能作为 TAC-style baseline 的数据锚点。它们的价值是暴露了三个可复现的数据问题：源身份不可追溯、固定 gain 不等于可听响度、人工 review 没有被机器门禁约束。下一步仍是 `docs/27_real_speech_sfx_evidence_pilot.md` 的 30 条 LibriSpeech + ESC-50 pilot；本轮只是把“源响度”和“渲染后可听性”补成硬门禁，不再制造 v3c。

在该 pilot 自动门禁和 30/30 人工听审通过前，不下载模型、不训练 B3，也不调 loss、prompt、EQ、压缩或 ducking。

## 2. 最新代码审计

### 2.1 `build_real_mix_v3.py` 是确定性失效，不应运行

`for i in range(n_mixtures)` 的循环体只有 scene、ID 和 duration 三行；从 `n_clip` 起的渲染逻辑错误地退回到循环外。因此脚本会循环选择 200 次、只渲染最后一次，却在末尾打印写入 200 条。即使修复缩进，它仍会随机裁切 LibriSpeech 波形而保留截断后的 transcript 描述、使用裸 `except`、硬编码 `/tmp`，且没有 stable source ID、source hash、source group、stem 或 deterministic replay。

这不是“参数可能不理想”，而是脚本执行语义错误；结果不能用于论文或训练。

### 2.2 v3b 已有数据的定量事实

仓库内 `data/derived/real_mix_v3b/manifest.jsonl` 可解析为 200 个 scene、538 个 source slot：

| 项目 | 数值 |
|---|---:|
| SFX slot | 463 |
| music slot | 49 |
| speech slot | 26 |
| stable source ID | 0/538 |
| source path | 0/538 |
| source group | 0/538 |
| source dataset/license | 0/538 |
| 实际 gain 记录 | 0/538 |

scene 分布中，speech 场景只有 `speech_with_sfx=19` 和 `speech_with_music=7`。manifest 无法回答 26 个 speech slot 来自多少录音，也无法验证人工听到的文件是否就是生成时的源。它不是 canonical `ManifestEntry`，因此不能做 source-disjoint split、波形哈希绑定、replay、stem-sum 或渲染后 stem 审计。

`review_10.csv` 的答案栏没有形成完成且可校验的结构化结果；`docs/18_real_mix_v3b_review_feedback.md` 只记录了少量非正式反馈，其中至少包括：带人声歌曲被误当纯音乐、标注有音乐但实际听不到、SFX 遮住 speech。10 条样本也不足以估计 12 类 scene 的失败率。

## 3. 为什么固定 gain 不能修复问题

`gain_db=-6` 只表示在原波形振幅上乘一个常数。两个文件原始 RMS 若相差 20 dB，使用相同 gain 后仍相差 20 dB；因此把 speech 设为 +3、SFX 设为 -6 并不能证明 speech 真正高 9 dB。

整段 RMS 也不够。ESC-50 可能只有一个短促事件、其余时间接近静音；若用包含静音的整段 RMS 归一化，会把短促事件峰值过度放大。为此本轮区分：

- `rms_dbfs`：整个抽样窗口的 RMS，用于静音和文件质量检查；
- `active_rms_dbfs`：仅在 `abs(sample) >= max(1.5/32768, 0.1 * global_rms)` 的样本上计算，用于混音响度归一化；
- `gain_db = sampled_target_active_rms - source_active_rms_dbfs`；若绝对增益超过 24 dB，直接失败而不是掩盖异常源。

pilot 的 active-RMS 目标是 speech -22~-20、SFX -31~-27、ambience -38~-34 dBFS。它们只是可证伪 pilot 的初始区间，不是论文最终分布；最终分布必须由真实短视频统计和更大规模人工听审校准。

## 4. 本轮代码闭环

1. `source_catalog.py` 在 prepare 阶段记录全段与 active RMS；catalog 仍由文件 SHA-256、内容 fingerprint、精确许可证和 source group 约束。
2. sampler 从 catalog 的 active RMS 反推 gain，把两个数据集放到可比较的绝对响度范围；manifest 同时冻结两个 RMS 和最终 gain。
3. clipping guard 对最终 mixture、dry mixture 和所有 stem 使用同一 master gain，并重新求和得到 exact dry mixture。
4. mixture audit 重新读取写盘后的 PCM stem，而不是相信 gain 元数据；检查 stem 是否存在、路径是否越界、采样率和 source ID 是否一致。
5. 对每类 stem 计算实际 active RMS；在 speech 与非 voice stem 真正重叠的样本上计算 `20 log10(RMS_speech/RMS_competitor)`。
6. 自动门禁要求所有 stem 可读、无 stem 低于 floor、至少 80% speech scene 可测 margin、低于 3 dB 的 speech scene 不超过 10%。失败 sample 会进入 `violation_samples` 和 `stem_audibility_violations`。

这些检查只证明“数据进入人工 review 前满足最低物理一致性”，不能替代人工判断语音是否可懂、caption/transcript 是否正确、事件时间是否合理。

## 5. 你需要做什么

严格执行 `docs/27_real_speech_sfx_evidence_pilot.md` 第 5 节，顺序不能互换：

1. 在服务器拉取代码并安装 `.[data,dev]`；确认 `sceneledger.__file__` 位于当前 checkout。runner 也会再次检查并把路径记录到 `sceneledger_import_path.txt`，防止旧 editable install 污染结果。
2. 使用新的 run 目录下载并重新 prepare ESC-50；戴耳机审核 20 条 source，完成并验证 source audit。旧 catalog 缺 `active_rms_dbfs`，不能复用。
3. 在同一新 run 目录重新 prepare Mini LibriSpeech + `test-clean`；戴耳机核对 30 条官方 transcript，完成并验证 source audit。
4. 生成 `pilot_test.yaml`，运行 `scripts/run_real_speech_sfx_pilot.sh`。输出目录必须是新目录，禁止覆盖旧结果。
5. 先看 `$RUN_ROOT/output/mixture_quality.json`。只要 `pass=false`，立即停止，不创建训练集；返回报告和 3–5 个失败 mixture/stem。
6. 自动门禁通过后，才填写 30 行 `human_audit_tasks.csv`，再运行 summary。

重点读取：

```text
failed_checks
metrics.stem_audibility.by_kind
metrics.stem_audibility.minimum_speech_competitor_margin_db
metrics.stem_audibility.speech_overlap_measured_fraction
stem_audibility_violations
```

### GO

自动门禁全部通过，30 行听审完整且无 uncertain，severe=0，总失败不超过 2/30，同一模板同一 criterion 的失败少于 2 次。此后才实现同一 scene plan 的 clean/RIR/echo/noise 单因素配对数据。

### NO-GO 定位

| 失败字段 | 只检查这一环 |
|---|---|
| `all_sources_have_provenance` | catalog prepare/审核绑定 |
| `stem_rms_floor_violation_fraction` | source active RMS、fade、master gain |
| `speech_overlap_measured_fraction` | placement 与真实 activity |
| `speech_competitor_margin_violation_fraction` | 相对响度目标 |
| replay/stem-sum | renderer 与 manifest |
| transcript 人工失败 | source 映射与非连续源裁切 |
| timestamp 人工失败 | VAD、fade、0.1 s activity |

一次只修一行，不扩大样本量，也不启动训练。

## 6. 下一轮是否需要新代码

当前 30 条 pilot 不再缺代码，缺的是服务器上的真实数据制品和 80 次人工判断（20 条 ESC source、30 条 LibriSpeech source、30 条 mixture）。在这些结果返回前继续增加模型代码没有可验证锚点。

若 GO，下一轮需要实现 corruption-pair builder，使同一 source/placement 只改变一个 RIR、echo 或 noise 条件，并冻结 clean/corrupted pair ID 与 stem 对齐。若 NO-GO，则只依据上表失败字段修改对应环节。
