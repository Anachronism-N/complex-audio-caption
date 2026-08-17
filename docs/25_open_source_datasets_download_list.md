# 开源数据集下载清单（2026-08-17 联网检索）

> 用途：支撑 `FileSourcePool` 的 5 类源（`speech`/`vocal`/`music`/`sfx`/`ambience`）+ 强时间标注 + 真实分轨（CARC）+ 真实 RIR/噪声。
> 关联：`docs/24`（流程盘点与扩充方案）、`docs/03 §3`（许可分层）、`MEMORY.md`（P0 优先级）。
> 检索日期 2026-08-17。链接为检索确认结果；**下载后务必 `sha256sum` 记指纹**（`docs/24` 第 0 步）。

## 0. 环境约定（复用 docs/24 §4.6）

```bash
export SRC_ROOT=/apdcephfs_gy4/share_302533218/cedricnie/complex-audio-caption/data/sources
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像
export HF_HOME=$SRC_ROOT/.hf_cache              # 禁止指向 /tmp
mkdir -p $SRC_ROOT
```

---

## 1. speech：多说话人 / 重叠 / 带转录

| 数据集 | 规模 | 关键价值 | 许可 | 下载链接 |
|---|---|---|---|---|
| **LibriSpeech** | 1000 h（train-clean-100 为 100 h 子集） | 干净单说话人 + verbatim 转录；多说话人靠混合构造 | CC BY 4.0 | 官网 `https://www.openslr.org/12/`；HF `https://huggingface.co/datasets/openslr/librispeech_asr`；ModelScope `https://www.modelscope.cn/datasets/pkufool/LibriSpeech` |
| **AMI Meeting Corpus** | 100 h 真实会议 | **真实自然重叠** + diarization + 转录 | CC BY 4.0 | `https://groups.inf.ed.ac.uk/ami/download/`（需注册） |
| **LibriCSS** | 10 h | LibriSpeech 重放录制，**重叠率 0%–40% 显式分档**，天然适配分层评测 | MIT | `https://github.com/chenzhuo1011/libri_css` |
| **AISHELL-4** | 120 h 中文会议，8 通道 | 中文 + 远场 + 重叠 | Apache 2.0 | `https://www.openslr.org/111/` |
| **AliMeeting** | 120 h 中文会议 | 中文远场多说话人 | 研究用 | `https://www.openslr.org/119/` |
| **CHiME-6** | 真实家庭聚会 | 多说话人 + 强噪声极端退化 | 研究用（需注册） | `https://chimechallenge.github.io/chime6/` |
| **VoxCeleb1/2** | 1251/6112 speakers | speaker identity 多样性极高（source embedding） | CC BY 4.0（需申请） | `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/` |
| **WenetSpeech** | 10000+ h 中文多域 | 大规模中文（可选） | 需申请 | `https://github.com/wenet-e2e/WenetSpeech` |

> P0 建议：**LibriSpeech train-clean-100**（零成本，已下载过）+ **AMI**（真实重叠）+ **LibriCSS**（重叠率分档）。

---

## 2. vocal：歌声 + 歌词对齐（补 `lys` 从 0 到有）

| 数据集 | 规模 | 关键价值 | 许可 | 下载链接 |
|---|---|---|---|---|
| **MUSDB18-HQ** | 150 首，vocals 独立分轨 | vocals 直接作 `vocal` 源；other 作纯器乐 `music` 源，**一份补两缺口** | CC BY-NC-SA 4.0 | Zenodo `https://zenodo.org/records/3338373`；HF `https://huggingface.co/datasets/salu133445/musdb18`；ModelScope `https://www.modelscope.cn/datasets/OmniData/MUSDB18` |
| **DALI v2** | 5358 首，音频-歌词-音符时间对齐（line+word 级） | 唯一大规模歌词时间戳；只发标注+YT ID，音频需自取 | 标注 CC BY-NC-SA | GitHub `https://github.com/gabolsgabs/DALI`；Zenodo `https://zenodo.org/records/2577915` |
| **jamendolyrics** | 79 首，word 级时间对齐，多语言 | 音频可直接下载（Jamendo CC），适合作 `lys` 评测金标准 | 各曲不同（CC） | GitHub `https://github.com/f90/jamendolyrics`；HF `https://huggingface.co/datasets/jamendolyrics/jamendolyrics` |
| **MoisesDB** | 240 首 / 45 艺术家，11 类细分 stem | 比 MUSDB18 更细分轨，vocals 单独 | 研究用（需注册） | HF `https://huggingface.co/datasets/wearemusicai/moisesdb`；官网 `https://music.ai/research/` |
| **MedleyDB** | 122+196 首多轨 + 旋律标注 | 含 vocal activation 标注 | CC BY-NC-SA | `https://medleydb.weebly.com/`（NYU MARL） |
| **MIR-1K** | 1000 片段，人声/伴奏左右声道分离 | 极易用：直接取左右声道 | 研究用 | `https://sites.google.com/site/unvoicedsoundseparation/mir-1k` |
| **Opencpop** | 100 首中文歌声，带音素时间标注 | 中文 `lys` | CC BY-NC-SA | GitHub `https://github.com/wenet-e2e/opencpop` |
| **M4Singer** | 多风格中文歌声，带对齐标注 | 中文 `lys` 扩量 | 研究用 | GitHub `https://github.com/M4Singer/M4Singer`；HF `https://huggingface.co/datasets/umoubuton/m4singer` |
| **OpenSinger** | 45.7 h 歌声 | 多歌手 | CC BY-NC-SA | 见 `open-mmlab/Amphion` 的 Sing-0.4k 清单 |

> P0 建议：**MUSDB18-HQ**（已下载，零成本）+ **jamendolyrics**（word 级对齐作金标准）。DALI 作 P1 扩量（注意音频合规）。

---

## 3. music：纯器乐（修 34.6% GTZAN 人声污染）

| 数据集 | 规模 | 关键价值 | 许可 | 下载链接 |
|---|---|---|---|---|
| **MUSDB18-HQ 的 other/drums/bass** | 150 首 | **保证零人声**（vocals 已剥离） | CC BY-NC-SA | 同 §2 |
| **Slakh2100** | 2100 首合成多轨（MIDI 渲染），145 h | 完美 MIDI 对齐时间标签 + 逐乐器分轨，CARC 真分轨监督理想源 | CC BY 4.0 | Zenodo `https://zenodo.org/records/4599666`；GitHub `https://github.com/ethman/Slakh` |
| **MTG-Jamendo** | 55000 首，含 instrumental 标签 | 可按 genre/instrument/mood 筛纯器乐 | CC | `https://github.com/MTG/mtg-jamendo-dataset` |
| **FMA (full/large)** | 106574 首 + 元数据 | 规模化 + genre 元数据 | 各曲不同（CC） | `https://github.com/mdeff/fma`（small 已下载，large 需申请） |
| **MedleyDB** | 含 instrumental-only 曲目 | 多轨 + 标注 | CC BY-NC-SA | 同 §2 |

> P0 建议：**MUSDB18-HQ accompaniment 替换 GTZAN**（这是 `docs/21` 错误结论的真正解法）。**Slakh2100** 紧随其后（RQ5 CARC 唯一现成真分轨源）。

---

## 4. sfx / ambience：扩类别 + 强时间标注

| 数据集 | 规模 | 关键价值 | 许可 | 下载链接 |
|---|---|---|---|---|
| **FSD50K** | 51197 条 / 108 h / 200 类 | 把 sfx 从 ~30 类扩到 200 类 | CC BY/CC0（逐条） | Zenodo `https://zenodo.org/records/4060432`；HF `https://huggingface.co/datasets/philgzl/fsd50k`；ModelScope `https://www.modelscope.cn/datasets/OmniData/FSD50K` |
| **AudioSet Strong** | 训练 103k + 评测 17k 片段，含 onset/offset | 唯一大规模强时间标注（~600 类）；音频需自取 YouTube | 标注 CC BY 4.0 | 标注 `https://research.google.com/audioset/download_strong.html` |
| **TACOS** | 12358 录音 / 47748 条强对齐自由文本 caption | **与本任务定义最接近的现成真实数据**；破自我验证闭环 | CC（Freesound 来源） | GitHub `https://github.com/OptimusPrimus/tacos`；Zenodo `https://zenodo.org/records/15379789`；论文页 `https://huggingface.co/papers/2505.07609` |
| **DESED** | 家庭环境 SED，含真实+合成强标注 | 家庭场景强标注，soundbank 可自行混合 | CC BY 4.0 | 官网 `https://project.inria.fr/desed/download/`；GitHub `https://github.com/turpaultn/DESED` |
| **FUSS** | 单源 + 混合，含 ground-truth stems | 现成"单源+混合"配对，适合 CARC | CC BY 4.0 | Zenodo `https://zenodo.org/records/3743844` |
| **UrbanSound8K** | 8732 条 / 10 类 | 城市声（已下载，只抽了 200 条） | CC BY-NC 3.0 | `https://urbansounddataset.weebly.com/urbansound8k.html`；HF `https://huggingface.co/datasets/danavery/urbansound8K` |
| **ESC-50** | 2000 条 / 50 类 / 5 s | 当前主 sfx 源（v7 用到 ~30 类） | CC BY-NC 3.0 | GitHub `https://github.com/karolpiczak/ESC-50` |
| **VGGSound** | 200k 片段 / 309 类，音视频对齐 | 未来 AV verifier | CC BY 4.0 | `https://www.robots.ox.ac.uk/~vgg/data/vggsound/` |
| **Clotho v2** | 6974 条 / 每条 5 句 caption | clip 级 caption 对照 | CC | Zenodo `https://zenodo.org/records/4783391` |
| **WavCaps** | ~400k 条弱标注 caption | 大规模 caption 预训练 | 各源不同 | `https://github.com/XinhaoMei/WavCaps` |
| **AudioTime**（PicoAudio 配套） | 带精确 onset/offset 的时间对齐 audio-text | 与本任务同源（时间可控生成） | 需核实 | GitHub `https://github.com/zeyuxie29/PicoAudio`；论文 `https://arxiv.org/abs/2407.02869` |

> P0 建议：**TACOS**（唯一现成"真实录音+自由文本+时间段绑定"，可直接作第二评测集）+ **FSD50K**（扩类别池）。

---

## 5. 真实 RIR 与噪声（替换合成衰减白噪声 RIR）

| 数据集 | 内容 | 下载链接 |
|---|---|---|
| **OpenSLR SLR28 (RIR_NOISES)** | 真实+模拟 RIR、等向噪声、点源噪声 | `https://www.openslr.org/28/`；HF `https://huggingface.co/datasets/schism-audio/openslr-rirs` |
| **MUSAN** | 音乐/语音/噪声三类 109 h | `https://www.openslr.org/17/` |
| **BUT ReverbDB** | 真实多房间 RIR + 背景噪声 | `https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database` |
| **MIT IR Survey** | 271 条真实环境 IR | `https://mcdermottlab.mit.edu/Reverb/IR_Survey.html` |
| **ACE Challenge** | 带 T60/DRR 标注的真实 RIR | `http://www.ee.ic.ac.uk/naylor/ACEweb/` |

> P0 建议：**SLR28 + MUSAN**（语音社区事实标准，`Conditions.t60_sec` 可从真实 RIR 反算）。

---

## 6. 一键下载脚本（示例）

```bash
export SRC_ROOT=/apdcephfs_gy4/share_302533218/cedricnie/complex-audio-caption/data/sources
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$SRC_ROOT/.hf_cache
mkdir -p $SRC_ROOT

# P0 三件套（已下载过可跳过，注意落持久路径）
huggingface-cli download salu133445/musdb18        --repo-type dataset --local-dir $SRC_ROOT/musdb18
huggingface-cli download openslr/librispeech_asr    --repo-type dataset \
  --include 'train-clean-100/*' --local-dir $SRC_ROOT/librispeech

# TACOS（Zenodo）
wget -P $SRC_ROOT/tacos https://zenodo.org/records/15379789/files/...  # 文件名以仓库为准
git clone https://github.com/OptimusPrimus/tacos.git $SRC_ROOT/tacos-repo

# FSD50K
wget -P $SRC_ROOT/fsd50k https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip

# Slakh2100（RQ5 CARC）
wget -P $SRC_ROOT/slakh https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz

# RIR + 噪声
wget -P $SRC_ROOT/rirs  https://www.openslr.org/resources/28/rirs_noises.zip
wget -P $SRC_ROOT/musan https://www.openslr.org/resources/17/musan.tar.gz

# 歌词对齐金标准
git clone https://github.com/f90/jamendolyrics.git $SRC_ROOT/jamendolyrics

# 下载后记指纹
find $SRC_ROOT -maxdepth 2 -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.tgz' \) \
  -exec sha256sum {} \; > $SRC_ROOT/SOURCES.sha256
```

---

## 7. 合规红线（复用 docs/03 §3）

| 数据集 | 红线 |
|---|---|
| MUSDB18 / MedleyDB / MoisesDB | **CC BY-NC**：可训练，**再分发受限**，benchmark 只能发 ID + 时间段 |
| DALI / AudioSet Strong | 只提供标注，**音频需自取 YouTube**，需许可分层 + tombstone 机制 |
| CHiME-6 / VoxCeleb / MoisesDB / AMI | 需注册或申请，**下载前先确认周期**，勿卡主线 |
| GTZAN | 无明确许可 + 含已知标注错误 → 本就应替换 |
| FSD50K / TACOS / FUSS | 逐条/来源不同许可，入 manifest 时记录 `license_tier` |

---

## 8. 优先级汇总（对齐 docs/24 §5 与 MEMORY.md 待办）

| 优先级 | 数据集 | 解锁的研究问题 |
|---|---|---|
| **P0** | MUSDB18-HQ | `lys` 从 0 到有 + 修 34.6% 音乐人声污染（一份补两缺口，已下载） |
| **P0** | LibriSpeech train-clean-100 | RQ1 多说话人 pointer + RQ6 cpWER 可测 |
| **P0** | TACOS | 唯一现成"真实录音+强对齐 caption"，破自我验证闭环 |
| P1 | FSD50K | sfx 类别 30 → 200 |
| P1 | SLR28 + MUSAN | 真实 RIR 替换合成 RIR |
| P1 | AMI / LibriCSS | 真实重叠（非人工混合） |
| P2 | Slakh2100 | RQ5 CARC 真分轨最小闭环 |
| P2 | jamendolyrics / DALI | `lys` 评测金标准 / 扩量 |

> 注意：AudioTime 的官方下载入口（是否独立发布 dataset 文件）本次检索未确认到直接 Zenodo/HF 链接，仅确认到 PicoAudio 仓库与论文页；使用前需在 `zeyuxie29/PicoAudio` 仓库 README 中核实数据获取方式。
