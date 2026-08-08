# 相关工作与研究空缺

> 调研冻结日期：2026-08-08。优先引用论文原文、官方项目页、会议论文集和数据集主页。2026 年预印本和匿名在审稿件尚未经过稳定的同行评审，文中将其结论标为“作者报告”，不把排行榜数字视作最终事实。

## 1. 问题重新定义

传统 Automated Audio Captioning (AAC) 多数把 10-30 s 音频压缩为一句 clip-level 描述，通常忽略逐字 speech、说话人、歌词和精确事件边界。这里需要的是一个更强任务：

- 输入：单通道或双通道、任意含 speech/music/general sound 的真实音频；
- 输出：一个可解析的统一 caption，显式区分 `<speech>`、`<music>`、`<lys>`、`<sfx>`；
- 结构：允许事件重叠、同一声源多段出现、多说话人和前景/背景关系；
- 时间：输出量化到 0.1 s，并报告真实定位误差；
- 条件：在混响、回声、噪声、压缩失真、音乐-人声混杂和多说话人重叠下保持可靠；
- 可靠性：既不能漏掉被掩蔽但仍可听的事件，也不能凭语言先验补出不存在的事件。

## 2. 最接近的工作

| 工作 | 已解决的关键部分 | 与目标任务的主要差距 |
|---|---|---|
| [TAC: Timestamped Audio Captioning](https://arxiv.org/html/2602.15766v1) | 直接生成带 `[music]/[sfx]/[speech]` 的 dense caption；可请求 0.1 s 分辨率；Dynamic Acoustic Mixer、时间 token 加权 loss、TAC-V | 依赖大量合成混音和许可单源数据；示例 scene template 明确避免 speech-speech overlap；speech 转录由外部 Whisper 后处理；无 lyrics 主任务；作者报告存在 sim-to-real 和细粒度音乐不足 |
| [TimeAudio](https://arxiv.org/html/2511.11039) | temporal markers、absolute time encoding、长音频 token merging；260k FTAR 覆盖 dense caption、grounding、speech timeline summary | 没有统一音乐/歌词/多说话人输出；dense caption 仍主要来自 AudioSet-Strong/TACOS/AudioTime；不是复杂混合鲁棒性方法 |
| [SpotSound](https://arxiv.org/html/2604.13023) | timestamp-interleaved audio tokens；正/负 query 训练抑制不存在事件幻觉；needle-in-a-haystack benchmark | Query-based 单事件/多实例定位，而非一次性全场景 caption；最终为效率采用 1 s 粒度；作者明确把 polyphony、瞬态事件和 repeated-instance 作为未解问题 |
| [TEMPO](https://openreview.net/forum?id=LoXjHBlPEd) | 一个 LALM 覆盖 multi-speaker ASR、diarization、audio grounding、dense AAC、timestamped music caption；atomic time tokens、time-aware projector、Gaussian loss、GRPO | 匿名 ACL ARR 在审稿；五任务由不同 prompt 区分，不是同一混合场景的一次输出；音乐来自 Slakh2100 合成 MIDI，未覆盖歌词与真实 music/speech/sfx 混合；119k 训练与 10k benchmark 的结论待评审 |
| [MOSS-Audio](https://arxiv.org/html/2606.01802v3) | 4B/8B 统一 speech、sound、music；12.5 Hz encoder、DeepStack、2 s time markers；分支标注含多说话人、music structure 和 lyrics | 重点是广义 audio understanding/ASR/QA；没有针对单个统一事件账本的 benchmark，也未给出混响、回声、强噪声和高并发声源下的系统鲁棒性分解 |
| [AudioChat](https://arxiv.org/html/2602.17097v1) | 6M agent 生成的多轮复杂 audio stories；含多说话人、前后景 sfx、panning/loudness；统一理解/生成/编辑 | 数据和评测主要是合成 story；StoryGen-Eval 仅 1,200 条且由同一 AudioCopilot 生成；模型/部分数据不发布；timestamped unified caption 不是核心评价 |
| [Audio-Omni](https://arxiv.org/html/2604.10708v2) | 使用冻结 MLLM + DiT 统一 general sound/music/speech 的理解、生成与编辑；AudioEdit 约 1.1M 对 | 研究中心是生成/编辑；没有事件级 0.1 s caption 目标或复杂声景 caption benchmark。其“真实数据 + SAM Audio 分离 + 合成数据”构建思想可借鉴 |
| [AudioCapBench](https://arxiv.org/abs/2602.23649) | 以 1,000 样本快速测 speech/sound/music caption，并显式评准确、完整和幻觉 | 没有细粒度时间、重叠声源和鲁棒性轴；作者报告现有模型在 music 上最弱，说明统一 caption 仍存在显著 domain gap |

### 2.1 TAC 给出的强基线与可攻击点

TAC 是必须正面超过的工作，而不是只在 related work 中引用。它已经做到：

- 使用 scene templates 和单源素材构造动态复调混音；
- 随机控制描述风格、merge threshold、activity threshold 和 0.01/0.1/0.5 s 等时间分辨率；
- 使用 atomic timestamp tokens 并提高时间 token 的交叉熵权重；
- 在 TACOS 上用 100 ms segment F1、event F1 与 FLAM-based hallucination 指标评价；
- 用 `[music]`、`[sfx]`、`[speech]` 等类型前缀输出可解析 caption。

但 TAC 论文也留下直接的研究空间：

1. 其最佳 TACOS event F1 约 0.50，说明即使 0.1 s 序列化可用，真实边界和实例匹配仍远未解决；
2. 0.01 s 输出反而使 event F1 下降且 hallucination 上升，证明更细 token 并不自动带来更准边界；
3. 训练过久出现 synthetic overfitting；移除 TACOS 的真实标注后性能明显下降；
4. 作者明确报告 dramatic-event prior、sim-to-real gap 和 chord 等音乐细节不足；
5. speech 内容由另一个模型补写，联合训练和联合错误传播没有被检验。

### 2.2 TEMPO 并没有“终结”本问题

公开索引显示 TEMPO 使用 68,456 条真实训练实例，并加入合成数据，总规模约 119k。真实部分包括 AMI/ICSI/Switchboard 的 multi-speaker ASR 与 diarization，AudioSet Strong 的 grounding，TACOS 的 dense caption，以及从 Slakh2100 MIDI 派生的 instrument/tempo/chord 音乐时间标注。其贡献是统一**任务接口**和 temporal post-training。

本项目要研究的是更难的联合分布：同一个真实录音中说话、歌唱、伴奏、sfx 和噪声同时存在，模型一次输出全部结果。TEMPO 的五类样本和五种 prompt 可以共享模型，却没有证据表明它学习了这种跨域并发、标签互斥/重叠关系，也没有真实 lyrics 和强退化条件。因此不能把“多任务统一”误写成“复杂混合场景统一”。

### 2.3 SpotSound 揭示了评价缺口

SpotSound 的核心发现很重要：如果只问模型“这个事件在哪里”，许多 LALM 在事件不存在时仍会强行输出时间段。它通过 positive/negative query 把“存在性判断”放在定位之前，并在 300 条平均约 53 s、目标覆盖约 8.4% 的真实录音上测试。其附录还指出，粗糙标注会惩罚更精细的多实例预测。

对本项目的启示是：

- 全量 caption 也必须有显式 no-event/null slots 和校准，而非总要填满若干事件；
- benchmark 要标注多段重复事件，不能把静音间隔粗暴并入一个长区间；
- 评价必须把“是否存在”“是什么”“何时发生”拆开；
- 0.1 s 标签要带边界不确定区间，否则精细模型会因标注噪声被错误惩罚。

## 3. 时间对齐与开放词汇检测

- [TACOS](https://arxiv.org/html/2505.07609) 提供 12,358 个 Freesound 录音和 47,748 条与具体时间段绑定的自由文本 strong captions，是训练开放词汇 temporal alignment 的核心真实数据；但语音转写被清除，且不以 music/lyrics/speaker 为中心。
- [FLAM](https://arxiv.org/html/2505.05335) 在 ICML 2025 提出 frame-wise language-audio modeling，以局部对比学习做 open-vocabulary SED，并结合合成强标注与 AudioSet Strong/DESED/UrbanSED。它适合作为局部证据验证器或初始化，而不是完整 captioner。
- [Detect Any Sound](https://arxiv.org/abs/2507.16343) 把 SED 视作多模态 query 的 frame retrieval，显式解耦事件识别和时间定位；可作为事件 slot 的开放词汇检测 teacher。
- [AudioSet Strong](https://research.google.com/audioset/download_strong.html) 对约 103k 训练片段和约 17k evaluation 片段提供 onset/offset，但采用约 600 类封闭 ontology，且所有音乐内部细节只标为 Music。
- [MixIT](https://papers.nips.cc/paper/2020/hash/28538c394c36e4d5ea8ff5ad60562a93-Abstract.html) 证明仅用真实 mixture-of-mixtures 可以做无监督/半监督分离和域适配。本项目借用其“混合代数”，但监督对象从 waveform decomposition 扩展为可解析事件集合。

## 4. Caption 数据与跨模态弱监督

| 数据/工作 | 规模与优势 | 对本任务的限制/用途 |
|---|---|---|
| [AudioCaps](https://audiocaps.github.io/) | 约 46k 人工 caption，来自 AudioSet | clip-level；通常不转写 speech；可做语言风格和全局 AAC 保持 |
| [Clotho](https://zenodo.org/records/3490684) | 4,981 音频、24,905 captions；每段 15-30 s | 明确移除 speech transcription；无时间边界 |
| [WavCaps](https://arxiv.org/abs/2303.17395) | 约 400k 弱标注 audio-caption；LLM 清洗网页描述 | 规模大但时间和真实性弱；适合预训练而非强监督 |
| [FSD50K](https://arxiv.org/abs/2010.00475) | 51,197 CC 音频、200 类，波形可再分发 | 多数不是强时间标注；适合单源/事件素材和合规发布 |
| [VggCaps / Multi2Cap](https://aclanthology.org/2025.emnlp-main.715/) | 用视频和 LLM 生成更丰富 captions；训练时 AV grounding、推理仅音频 | 证明 video 可作为 privileged modality；但仍以全局 caption 质量为主 |
| [SAM Audio](https://arxiv.org/abs/2512.18099) | text/visual/time-span prompt 的统一 speech/music/sound 分离 | 可作为真实数据的 pseudo-stem teacher；分离结果不是 ground truth，必须验证和保留置信度 |
| [AudioScope](https://research.google/pubs/self-supervised-audio-visual-separation-of-on-screen-sounds-from-unlabeled-videos/) | 从无标注 in-the-wild 视频自监督分离 on-screen sound | 支持用短视频做弱监督；同时提醒“画面中可见”不等于“声音存在” |

视频只能作为候选证据。若 VLM 看见狗就直接写入 barking，会制造典型 cross-modal hallucination。每个视觉候选仍须通过局部 audio-text 相似度、声源分离后的可听性或同步性检验；视觉单独出现不能成为正标签。

## 5. Speech、重叠说话与恶劣声学条件

- [WHAMR!](https://arxiv.org/abs/1910.10279) 专门研究单通道、噪声和混响条件的两说话人分离，适合构造可控 SNR/T60 压力测试。
- [LibriCSS](https://arxiv.org/abs/2001.11482) 是 far-field 重放得到的连续多说话人数据，覆盖不同重叠率，适合 speaker-attributed transcription 的真实评测。
- AMI、ICSI、Switchboard 是 TEMPO 使用的说话人标注语料，可用于 diarization/ASR 预训练，但它们没有复杂 music/sfx 联合 caption。
- 多说话人输出不能只用普通 WER；应报告 permutation-aware cpWER/tcpWER、DER/JER，并单独评价重叠区。

## 6. Music 与歌词

- [DALI](https://transactions.ismir.net/articles/10.5334/tismir.30) 提供 5,358（v1）/7,756（v2）首歌的多粒度时间对齐 lyrics 和 notes，可用于 `<lys>` 的 word/line alignment；需遵守其音频匹配与版权约束。
- [MUSDB18 lyrics extension](https://zenodo.org/records/3989267) 同时提供 vocals/bass/drums/other/mixture 和 lyrics 扩展，适合 music-vocal 干扰、stem-level 训练与受控重混。
- TEMPO 的 Slakh2100 音乐目标强调 MIDI 可导出的 chord/tempo/instrument；它不能代表真实录音中的演唱、混音、母带和歌词识别。
- MOSS-Audio 已使用 lyrics ASR、Chordino、BeatNet、madmom、Essentia、JukeMIR、SongFormer 等专家工具融合音乐 caption。因此本项目不应把“调用多个 MIR 工具”包装为主要创新；创新应在跨域联合事件账本、可信证据和真实复杂场景鲁棒性上。

## 7. 尚未被充分解决的空缺

1. **统一任务而非多任务菜单**：现有系统常在 ASR、music caption、SED 间切 prompt；少有一个输出同时覆盖同一 mixture 中的全部类型。
2. **物理证据约束**：大多数 LALM 直接对文本 token 做 CE，时间与语义都可能由语言先验主导；缺少“每个 event phrase 必须指向局部可听证据”的结构。
3. **真实复杂域适配**：TAC/FLAM/SpotSound/AudioChat 大量使用 programmatic mixtures；MOSS-Audio 使用真实自动标注，但没有反事实配对来识别 teacher 错误和 nuisance invariance。
4. **lyrics 与 overlapping speech**：环境音 caption、ASR、歌词识别和 music description 长期分别建模；它们的竞争与共存缺少统一 benchmark。
5. **0.1 s 评价可信度**：输出 token 粒度、encoder 帧率、人工边界精度和 perceptual onset 并非同一概念。当前论文常报告 IoU 或宽 collar，不能证明 100 ms 级准确。
6. **鲁棒性曲线**：多数工作报告一个综合分数，少有按 SNR、T60、echo delay、codec、并发源数、事件持续时间和前后景强度系统分解。
7. **可校准的“不知道”**：对不可听歌词、完全掩蔽 speech 或不确定声源，可靠模型应降低置信或只输出高层描述，而不是强行转写。

这些空缺共同支持下一篇论文的中心论点：**复杂音频 caption 的瓶颈不再只是语言模型的时间表示，而是如何建立可验证、可组合、抗声学扰动的事件证据。**

