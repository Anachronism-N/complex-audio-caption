# B3 第一步：真实音源池验收

当前只执行这一阶段。`source_readiness_report.json` 未通过前，不渲染、不导出 SFT、不训练模型。
旧 synthetic B2/B3 分数不能代替真实音源池验收。

## 1. 输入契约

把多个公开或内部授权数据集整理为 CSV/JSONL。每行必须包含：

- `path`：可读取的单源音频；
- `kind`：`speech/vocal/music/sfx/ambience` 之一；
- `text`：音频中实际可听内容的人工或数据集原生标签；
- `source_group`：说话人、歌曲、原视频或原始录音 ID；
- `license`：明确许可证或 `internal-authorized`，不得为空；
- vocal 额外要求 `verbatim=true`，且 `text` 必须是实际歌词；
- 建议填写 `dataset/language/identity`。

互联网无标注视频不能直接进入这个监督池。它们必须先完成授权、去重、切分和人工/teacher 审核，
并保留原视频 ID 作为 `source_group`。

## 2. 只运行 sources gate

第一次在服务器上只运行：

```bash
SOURCE_CATALOG=/data/b3/source_catalog.csv \
SOURCE_AUDIO_ROOT=/data/b3/audio \
SOURCE_PROFILE=smoke \
WORK_DIR=/data/runs/b3_smoke \
N_SAMPLES=100 \
STAGE=sources \
bash scripts/run_b3_data.sh
```

source gate 与模型锚点相互独立，因此这一阶段不要求 TAG summary。进入 `render` 时 runner 才会
强制 `TAG_SUMMARY` 已通过。

该命令先规范化 catalog，再逐文件解码并生成：

- `source_catalog.jsonl`：绝对路径和规范 metadata；
- `source_catalog_report.json`：catalog 级统计和哈希；
- `source_inventory.jsonl`：byte/decoded SHA-256、时长、采样率、声道、RMS、peak、削波率；
- `source_readiness_report.json`：所有检查、配额和稳定 `source_pool_id`。

修改 metadata 后只重跑 catalog+audit；只替换音频文件时可执行：

```bash
SOURCE_PROFILE=smoke WORK_DIR=/data/runs/b3_smoke \
STAGE=source-audit bash scripts/run_b3_data.sh
```

## 3. 自动验收内容

配置在 `configs/data/source_readiness.yaml`。smoke 和 release 使用不同的、版本化的配额。gate 会
拒绝：

- 缺少任一声音类型或 source group 数量不足；
- 文件不存在、无法解码、空音频、非有限值、近静音、严重削波或时长越界；
- 两个不同路径解码后得到相同音频；
- 缺失 license；
- vocal 没有逐字真实歌词；
- 任一 source 数量、group 数量或总时长低于 profile 要求。

查看失败项：

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('/data/runs/b3_smoke/source_readiness_report.json').read_text())
print('pass:', p['pass'])
print('source_pool_id:', p['source_pool_id'])
for check in p['checks']:
    if not check['pass']:
        print(check['name'], check['detail'])
PY
```

## 4. 进入下一环的条件

只有同时满足以下条件，才运行 `STAGE=render`：

1. `source_readiness_report.json: pass=true`；
2. `failed_checks=[]`；
3. 五类 source 和 source group 均达到 `smoke` 配额；
4. 对每类至少随机试听 10 条，确认 `text` 与真实声音一致；
5. 对所有 vocal 抽查歌词确实逐字可听；
6. 记录并冻结 `source_pool_id`。

render runner 会再次调用 readiness gate；报告缺失、profile 不一致或未通过时会立即停止。smoke
完成后才讨论 renderer 问题，smoke render/export/audit 全部通过后才建立新的 `release` 目录。

## 5. 当前真实状态

代码与 CPU fixture 已就绪，但仓库中没有服务器生成的真实 `source_readiness_report.json`，因此
当前状态仍是“音源池尚未复现”，不是“训练数据已准备好”。在该文件回传前，暂停 B2/B3/S1
结构实验。
