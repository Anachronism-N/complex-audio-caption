Exit code: 0
Wall time: 6 seconds
Output:
# 训练课程、偏好对齐与可验证奖励

> 核心原则：先让模型学会结构化检测和局部证据，再训练文字；先做 SFT 和反事实偏好，再考虑 RL。RL 不是时间戳准确性的起点，也不能用一个 LLM judge 替代声学证据。

## 1. 输出表示与训练边界

SceneLedger 同时保留两个输出空间：

1. 连续/密集输出：track/event presence、100 ms activity、TF mask、boundary distribution、track pointer；
2. 离散输出：每个 event 的 `<speech>/<lys>/<music>/<sfx>` 文本及最终 canonical serialization。

基线 B2 复现 TAC 的 atomic timestamp token；主模型不让语言模型单独承担边界回归。时间来自 activity/boundary head，序列化器把它量化到 0.1 s。这样 RL 只优化可采样的离散选择和序列，连续边界头继续使用可微监督与一致性损失。

## 2. 分阶段课程

### Stage T0：协议与冻结基线

- 冻结数据 schema、parser、deterministic serializer 和 matcher；
- 跑通 MOSS zero-shot、TAC-style autoregressive baseline；
- 固定 200 条人工开发集，所有结构变更都需重新跑兼容测试；
- 不进行 RL。

### Stage T1：时间与活动预训练

数据使用 Level A/B/C 中有精确 activity 的样本。冻结 MOSS decoder，训练 temporal fusion、track/event queries 与 activity/boundary heads：

$$
L_{T1}=L_{presence}+L_{activity}+L_{boundary}+L_{type}+L_{count}.
$$

目标是先回答“有几个来源、什么时候活跃”，不同时学习长文本。验收指标是 event F1、segment mAP、boundary MAE 和 source-count MAE。

### Stage T2：Track identity 与 event pointer

加入 track slots、source embedding、TF evidence 和 event-to-track pointer：

$$
L_{T2}=L_{T1}+L_{track\_match}+L_{pointer}+L_{contain}+L_{mask}+L_{diversity}.
$$

同一 speaker/singer 的多个 events 共享 track；不同 speaker 即使同时说话也必须分开。`L_diversity` 只防止 slot collapse，不能强制所有 slots 互斥，因为真实声音会重叠。

### Stage T3：局部 evidence caption SFT

先只对 oracle/matched event slots teacher-forcing，再逐步混入模型预测 slots：

- 30% steps：oracle track + oracle event spans；
- 30% steps：predicted track + oracle event matching；
- 40% steps：完全 predicted slots，并对 unmatched/no-event 训练 abstain。

使用 type-specific adapters：speech/lyrics 偏转录，music/sfx 偏描述。局部 prefix 是必要输入；global prefix 有门控和 token 上限，避免 decoder 绕过局部证据，仅凭全局语境讲故事。

### Stage T4：CARC 一致性训练

同一 batch 同时输入原始 `x` 与干预后 `x'`，共享模型参数并跨样本匹配 events。加入 add/remove/shift/preserve/audibility 损失。该阶段主要使用真实背景 Exact-CARC，缩小 TAC++ 到真实分布的差距。

### Stage T5：Counterfactual preference alignment

从真值或高置信 ledger 自动构造 chosen/rejected，不立即上 RL。优先尝试 DPO/IPO，因为稳定、可审计，并能直接利用成对 hard negatives。

### Stage T6：Verifiable GRPO

只有当 SFT 模型能稳定输出合法 schema、验证器在人工集上与人类相关、DPO 已有明确收益时启用。GRPO 用于优化离散 event selection、文本和序列化，不替代连续时间监督。

### Stage T7：验证结果蒸馏与校准

teacher-agent/audio-verified 模式产生 accepted/corrected outputs，蒸馏回 `student-fast`。最后用独立 calibration split 做 temperature scaling 或 isotonic regression，确定 accept/abstain 阈值。

## 3. Slot 匹配与监督损失

### 3.1 Track matching

对预测 track `i` 和目标 track `j` 定义成本：

$$
C^T_{ij}=\lambda_p C_{presence}+\lambda_y C_{type}+\lambda_a(1-\operatorname{softIoU}(a_i,a_j))
+\lambda_m C_{mask}+\lambda_s C_{source}.
$$

用 Hungarian algorithm 求 permutation-invariant assignment。无匹配 slot 监督为 null；residual/background slot 不参与普通 source-count loss。

### 3.2 Event matching

在 matched track 条件下计算：

$$
C^E_{ij}=\mu_y C_{type}+\mu_q C_{pointer}+\mu_a(1-\operatorname{softIoU})
+\mu_b C_{boundary}+\mu_t C_{text}.
$$

`C_text` 训练早期只用冻结 text embedding cosine，避免长 caption loss 主导匹配；assignment 确定后才计算 token CE。

### 3.3 Activity 与 boundary

- activity：focal BCE + soft Dice，缓解大部分时间为负类；
- boundary：预测 onset/offset 离散分布或 Gaussian mean/log-variance；
- multi-span event：activity mask 表达所有区间，boundary head 对每个 connected component 预测；
- uncertainty：以标签容差训练 heteroscedastic NLL，模糊混响尾部不强制单点。

例如 Gaussian boundary loss：

$$
L_b=\frac{(t-\mu)^2}{2\exp(2s)}+s,
$$

其中 `s=log σ`。必须限制 `σ` 范围并额外做 calibration，否则模型可能通过无限增大不确定性逃避误差。

### 3.4 Evidence grounding

每个生成 event 应满足：

- `inside_support`：文本与 event mask 内局部音频相似；
- `outside_margin`：该文本在局部证据上的支持高于非相邻区域；
- `target_residual_margin`：显式 track teacher 中，caption 对 target 的支持高于 residual；
- `counterfactual_delta`：移除该来源后，文本支持与 event confidence 应下降。

实现时可以用冻结 FLAM/CLAP/open-vocabulary grounding 分数作为弱监督，同时训练一个小型 evidence critic。critic 的训练正例来自真 stems/Exact-CARC，负例来自 time shift、wrong source 和 plausible-but-absent captions。

## 4. Hard-negative taxonomy

参考 AHA 的 counterfactual preference alignment 思路，并针对统一 ledger 扩展：

| 类别 | rejected 构造 | 要抑制的错误 |
|---|---|---|
| Omission | 删除弱但可听 event | 强源遮盖弱源导致漏检 |
| Insertion | 从同场景 ontology 加不存在事件 | 语言先验幻觉 |
| Identity swap | 交换 speaker/singer/track pointer | 多人归属错误 |
| Type swap | `<speech>`↔`<lys>`、`<music>`↔`<sfx>` | 模态混淆 |
| Time shift | 边界平移 0.2–2.0 s | 合理文本、错误时间 |
| Overlong span | 扩展到整段 | 用宽时间段骗取 IoU |
| Fragment/merge | 拆分或合并重复事件 | event 粒度错误 |
| Transcript corruption | 改实体、数字、否定词 | speech 内容幻觉 |
| Visual lure | 加入画面可见但未发声物体 | VLM 视觉偏置 |
| Echo duplication | 把回声标成第二 speaker/event | 复杂声学误归因 |

rejected 必须在语言流畅度上尽量接近 chosen；否则模型只学会识别坏文风。时间错误采用多个难度桶，不能都偏移 5 s 形成简单捷径。

## 5. DPO/IPO 设计

对输入音频 `x`、chosen ledger `y+`、rejected ledger `y-`：

$$
L_{DPO}=-\log \sigma\left(\beta\left[
\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)}-
\log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)}
\right]\right).
$$

实际训练分两种粒度：

- event-local pair：只改变一个 event，定位清楚，数据量大；
- ledger-global pair：改变计数、顺序、track assignment 或关系，贴近最终输出。

每个 batch 保留一定比例 SFT loss，防止 preference optimization 损害转录和格式。按 negative taxonomy 分层采样并分别报告收益。

## 6. 可验证 GRPO 奖励

### 6.1 为什么不使用单一 LLM judge

LLM 可以判断文本通顺和部分语义，但无法可靠听出 0.1 s 边界、弱音、speaker identity，也可能偏爱冗长回答。主奖励必须由 parser、匹配器、精确干预和音频 evidence 计算；LLM judge 仅作低权重语言质量项或离线分析。

### 6.2 完整监督样本奖励

对采样输出 `ŷ` 解析后与目标 `y` 匹配：

$$
R = w_fR_{format}+w_sR_{set}+w_tR_{time}+w_eR_{evidence}
+w_cR_{content}+w_qR_{track}+w_kR_{cal}-P_{gaming}.
$$

- `R_format`：schema 合法、ID 唯一、时间范围合法、排序正确；
- `R_set`：type-aware event F1，而非只奖励 precision；
- `R_time`：matched event 的 tIoU、boundary tolerance 和 shift error；
- `R_evidence`：局部支持、inside/outside margin、counterfactual drop；
- `R_content`：speech WER/实体准确、lyrics line error、music/sfx semantic score；
- `R_track`：speaker/singer/track pointer 与跨 event identity；
- `R_cal`：正确事件高置信、错误/模糊事件低置信或 abstain；
- `P_gaming`：过长 spans、重复 event、无意义极短 event、复制 prompt、空答案。

建议先把每项归一到 `[0,1]`，再固定权重；训练日志必须保存各分项，不能只看总 reward。

### 6.3 部分监督和 CARC 样本奖励

真实背景 `x` 没有完整标签时，不能把未标事件当 false positive。对 `(x, x')` 联合采样，只计算差分：

```python
r_add = match_injected_event(pred_after - pred_before, target_event)
r_preserve = agreement(non_target(pred_before), non_target(pred_after))
r_shift = equivariance(pred_shifted, delta)
r_remove = disappearance(pred_mixed, pred_clean, target_event)
r_audio = evidence_drop_after_removal(target_caption)
reward = weighted_sum(r_add, r_preserve, r_shift, r_remove, r_audio)
```

这是 CARC 最关键的可验证 reward：不要求 LLM 猜出真实背景所有声音。

### 6.4 GRPO 训练步骤

```python
for batch in loader:
    candidates = policy.generate(batch.audio, n=G, constrained_schema=True)
    rewards = [verifiable_reward(c, batch) for c in candidates]
    advantages = group_normalize(rewards)
    loss_pg = clipped_policy_loss(policy, ref_policy, candidates, advantages)
    loss = loss_pg + beta_kl * kl_to_reference + alpha_sft * sft_anchor
    optimize(loss)
```

推荐从 `G=4` 开始；若同组 reward 方差接近零，跳过或补充 hard prompt。持续监控 KL、输出长度、event count、各 type recall 和 abstain rate。

## 7. 防止 reward hacking

| 作弊行为 | 原因 | 防护 |
|---|---|---|
| 只输出强事件 | precision 奖励高 | 使用 event F1/coverage 和按 audibility 分层 recall |
| 把 span 覆盖整段 | 更容易命中 | boundary penalty、span length prior、on/off tolerance |
| 拆成很多相似事件 | recall 假增 | one-to-one matching、duplicate penalty、count error |
| 所有不确定项都 abstain | 避免错误 | selective risk-coverage 曲线和 coverage target |
| 复制 ASR 为 `<sfx>` | 文本相似度漏洞 | type-aware expert reward 与 track pointer |
| 根据视频补不可听事件 | VLM reward 偏置 | audio gate；audio 反证优先 |
| 输出固定模板 | 格式奖励过强 | format reward 封顶且权重低 |

每 1k–5k steps 在人工 dev set 上检查真实指标，而不是只看训练 reward。若 reward 上升而 hallucination/coverage 恶化，立即停止该阶段。

## 8. 幻觉抑制的因果测试

模型的 hallucination 不是只靠“输出更短”来衡量。至少做：

1. `source removal test`：移除目标来源后，相应 event confidence 应显著下降；
2. `wrong-window test`：把同一 caption query 放到非活动窗口，支持分数应下降；
3. `visual lure test`：画面出现但音频无声的物体不应生成 sfx；
4. `language-prior test`：常见共现声音缺失时不能脑补，如“车流”不必然有鸣笛；
5. `echo test`：原声与回声应保持同一 source identity，除非确有第二来源；
6. `low-audibility test`：在阈值附近允许 abstain，并报告 risk-coverage。

主指标是 counterfactual sensitivity/specificity 与 event-level hallucination rate，不只使用 caption LLM score。

## 9. 训练配置建议

MVP 可使用如下初值，最终需以 pilot 调参：

```yaml
model:
  track_slots: 8
  event_slots: 24
  time_resolution_sec: 0.1
  local_prefix_tokens: 4
  global_prefix_tokens: 2
training:
  stage_t1_steps: 20000
  stage_t2_steps: 30000
  stage_t3_steps: 50000
  stage_t4_steps: 30000
  pseudo_label_max_weight: 0.7
  exact_carc_weight: 1.0
preference:
  enabled_after_sft: true
  beta: 0.1
  sft_anchor_weight: 0.2
rl:
  enabled: false       # pilot 默认关闭；通过 gate 后开启
  group_size: 4
  kl_beta: 0.02
  max_span_ratio: 0.8
```

Gate 条件：B3/S2 在人工 dev 上格式成功率 >99%、event F1 稳定、验证 reward 与人工偏好 Spearman 相关显著，才把 `rl.enabled` 改为 true。

## 10. 必须完成的消融

- no track slots / flat event slots；
- no local evidence，只用全局 audio tokens；
- timestamp tokens vs boundary head vs hybrid；
- SFT only vs DPO vs GRPO；
- random negatives vs counterfactual hard negatives；
- pseudo-label SFT vs Exact-CARC delta supervision；
- LLM judge reward vs verifiable reward；
- audio-only verifier vs audio+VLM；
- fixed source count vs learned null/presence slots；
- deterministic serializer vs unconstrained LLM rewrite。

只有证明 `track identity + local evidence + counterfactual supervision` 三者各自有独立增益，论文的中心论点才成立。

## 11. 前置方法

- AHA 的反事实 hard-negative preference alignment 为 hallucination taxonomy 和 DPO 数据提供直接参考：[paper](https://arxiv.org/abs/2512.24052)。
- TEMPO 将 temporal SFT 与带时间 reward 的 GRPO 结合；截至本文冻结日其 OpenReview 版本仍应标记为匿名在审工作：[OpenReview](https://openreview.net/forum?id=LoXjHBlPEd)。
- TAC 的 atomic timestamp tokens 与 time-weighted CE 是 B2 的直接基线：[paper](https://arxiv.org/abs/2602.15766)。


