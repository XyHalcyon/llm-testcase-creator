# DeepSeek-V3.2 12K 混合序列数据集生成器

本目录用于基于 GSM8K 问题生成 DeepSeek-V3.2 性能压测数据集。数据集包含 10 档输入/输出长度，整体平均输入约 `12349 tokens`、平均目标输出约 `2056 tokens`，默认构造约 `25%` 的批内共享前缀。

生成器支持：

- EvalScope OpenAI Chat Completions 数据格式。
- AISBench `qa` 自定义数据格式。
- 每条请求独立的输出长度 `max_tokens`。
- KV block 对齐的共享前缀。
- 每轮压测自动生成不同内容，隔离上一轮留下的 Prefix Cache。

## 目录结构

```text
ds-v3.2-12k/
├── GSM8K.jsonl                 # 基础 GSM8K 数据，每行包含 question 和 answer
├── gen_data_12k.py             # 12K 混合序列数据集生成器
├── evalscope_mixed_plugin.py   # EvalScope 自定义数据集插件
└── README.md                   # 本说明文件
```

执行 `--format evalscope` 后，输出目录还会生成：

```text
├── gsm8k_12k_*.jsonl           # 生成的压测数据集
├── evalscope_mixed_plugin.py   # 自定义数据集插件
└── run_perf.py                 # EvalScope 压测启动模板
```

## 长度分布

| 输入 token | 输出 token | 占比 |
|---:|---:|---:|
| 55 | 10 | 0.56% |
| 330 | 55 | 0.34% |
| 660 | 110 | 1.01% |
| 1,320 | 220 | 9.00% |
| 2,630 | 440 | 12.94% |
| 5,266 | 870 | 15.69% |
| 10,500 | 1,760 | 28.35% |
| 21,060 | 3,500 | 26.55% |
| 42,100 | 7,000 | 5.23% |
| 84,200 | 14,050 | 0.34% |

脚本使用最大余数法分配整数条数，最终请求数严格等于 `--num-requests`。

例如 `--num-requests 3660` 时，各档数量依次为：

```text
21, 12, 37, 329, 474, 574, 1038, 972, 191, 12
```

## 环境要求

推荐使用 Python 3.10 及以上版本，并使用 DeepSeek-V3.2 自身的 tokenizer。

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install transformers tiktoken datasets
```

使用 EvalScope 时安装：

```bash
python -m pip install 'evalscope[perf]==1.9.1'
```

只加载 tokenizer 时出现以下提示不影响数据生成：

```text
PyTorch was not found. Models won't be available and only tokenizers ... can be used.
```

## 快速生成 EvalScope 数据集

进入目录并激活现有 EvalScope 环境：

```bash
cd "/Users/gaosir/Downloads/工作/数据生成脚本/dsv3.2/12k/ds-v3.2-12k"
source "../../.venv-evalscope/bin/activate"
```

推荐命令：

```bash
python gen_data_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../ds-v3.2-w8a8" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --format evalscope
```

不指定 `--output` 时，脚本会自动生成唯一文件名，例如：

```text
gsm8k_12k_c128_cache25_20260722_120000_a1b2c3d4.jsonl
```

`--concurrency` 不参与数据内容生成，只影响：

- 默认文件名中的 `c128`。
- 自动生成的 `run_perf.py` 中的 `parallel`。

## 保证两次生成的数据不同

`gen_data_12k.py` 已加入与 3K 生成器相同的批次隔离逻辑。

每次不指定 `--body-salt` 时，脚本自动生成唯一盐值：

```text
年月日_时分秒_随机十六进制
```

该盐值用于对整批内容进行重新组织，而不只是修改开头的一段文本。

### 1. 每批重新排列 GSM8K 语料

脚本使用 `SHA256(seed:body_salt)` 初始化独立随机顺序，对 GSM8K 问题重新排列，再进行 tokenizer 编码。因此两批数据的正文问题顺序不同。

### 2. 共享区每个块加入批次标记

使用 `--kv-block-size 128` 时，共享区按 128 token 组织，每个块开头加入：

```text
[shared-批次标识-块编号]
```

同一批次的所有请求使用相同共享块，因此仍然产生目标约 25% 的批内共享前缀；不同批次的每个共享块都带不同批次标识。

### 3. 非共享区每个块加入唯一标记

25% 共享边界后的正文同样按 128 token 组织，每个块加入：

```text
[body-批次标识-请求编号-块编号]
```

标记同时包含批次、请求和块编号，因此不同批次、不同请求以及同一请求的不同正文块都不会生成相同的完整标记块。

### 4. 共享区和正文使用不同起点

脚本根据批次摘要分别计算共享语料偏移和正文语料偏移，并结合请求编号选择正文起点。即使两轮都使用同一个 `GSM8K.jsonl`，生成的长文本排列也不同。

### 5. 唯一输出文件名

未指定 `--output` 时，每轮都会生成不同名称的 JSONL 文件，不会覆盖上一轮数据。

这些标记包含在原计划输入 token 长度中，不会额外增加输入长度。两次默认执行具有：

- 不同 GSM8K 问题排列顺序。
- 不同共享语料和正文偏移。
- 不同共享块标记。
- 不同正文块标记。
- 不同完整请求文本。
- 不同输出文件名和文件 SHA-256。
- 相同长度档、请求数量、输出长度分布和目标 25% 批内共享比例。

测试使用相同 seed、不同 salt 各生成 100 条 12K 数据，并对全部请求重新 tokenize 后按 128 token 切块，结果为：

```text
完整请求文本交集 = 0
跨批完整128-token块交集 = 0
max_tokens分布一致 = True
```

这里保证的是跨批次没有相同的完整 128-token 数据块。由于两批仍使用同一个 GSM8K 基础文件，单词、短句或问题片段可能相同，但它们不会以相同批次标记、相同正文顺序和相同完整块的形式出现。

注意：标记位于用户消息内容中。如果服务端在用户内容前插入超过一个完整 KV block 的固定 Chat Template 或系统提示，这些固定模板块仍可能跨批次命中。需要严格隔离整个服务端 Prompt 时，应使用服务端原生请求级缓存盐值，或在每轮压测前清理 Prefix Cache。

## 手动指定批次标识

通常不需要设置 `--body-salt`，让脚本自动生成即可。如果需要明确控制文件名和批次标识：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_${RANDOM}_${RANDOM}"

python gen_data_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../ds-v3.2-w8a8" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --format evalscope \
  --body-salt "${RUN_ID}" \
  --seed 42 \
  --output "./mix_12k_${RUN_ID}.jsonl"
```

固定所有参数、`--body-salt` 和 `--seed` 可以复现相同数据；改变 `--body-salt` 可得到不同批次。

## 指定固定输出路径

```bash
python gen_data_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../ds-v3.2-w8a8" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --format evalscope \
  --output "./mix_12k_unique.jsonl"
```

即使固定输出路径，每次自动生成的内容仍然不同，但旧文件会被覆盖。需要保留多轮结果时应省略 `--output` 或使用不同文件名。

## 输出数据格式

### EvalScope

每行是一个完整 Chat Completions 请求：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "生成的12K混合序列文本"
    }
  ],
  "max_tokens": 1760,
  "ignore_eos": true,
  "stream": true
}
```

每条请求携带自己的 `max_tokens`。生成的 `run_perf.py` 中必须保持：

```python
max_tokens=None
```

否则 EvalScope 全局输出长度可能覆盖数据集中的逐请求长度。

`evalscope_mixed_plugin.py` 会注册名为 `mixed` 的自定义数据集，同时兼容旧的 `prompt` 格式并自动转换为 `messages`。

### AISBench

每行格式：

```json
{
  "question": "生成的12K混合序列文本",
  "answer": "none",
  "max_tokens": 1760
}
```

生成命令：

```bash
python gen_data_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../ds-v3.2-w8a8" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --format aisbench
```

## 使用 EvalScope 压测

生成数据后修改自动生成的 `run_perf.py`：

```python
import os

import evalscope_mixed_plugin
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="deepseek-v3.2",
    api="openai",
    url="https://your-gateway.example.com/v1/chat/completions",
    api_key=os.environ["OPENAI_API_KEY"],
    dataset="mixed",
    dataset_path="/absolute/path/to/generated.jsonl",
    tokenizer_path="/absolute/path/to/ds-v3.2-w8a8",
    number=1000,
    parallel=128,
    rate=3,
    max_tokens=None,
    stream=True,
    name="mix_perf",
)

if __name__ == "__main__":
    run_perf_benchmark(args)
```

运行：

```bash
export OPENAI_API_KEY="你的API_KEY"
python run_perf.py
```

建议先使用小规模配置验证服务和请求格式：

```python
number=100
parallel=32
rate=1
```

确认无失败后再逐步增加。

参数含义：

- `number`：本轮发送的请求数。
- `parallel`：最大并发或在途请求数。
- `rate=-1`：闭环并发模式。
- `rate=N`：按约 `N req/s` 发送请求，`parallel` 作为在途请求上限。
- `max_tokens=None`：使用每条数据自己的输出长度，不能改成固定数字。

12K 数据中长输出请求较多，最大输出达到 `14050 tokens`。设置请求速率时，需要同时考虑长请求带来的连接占用时间，避免使用过大的 `parallel` 和 `rate` 导致网关主动断开连接。

## 使用 AISBench 压测

```bash
ais_bench --models <your_model> \
  --custom-dataset-path <generated.jsonl> \
  --custom-dataset-data-type qa \
  --max-out-len -1
```

`--max-out-len -1` 表示使用每条数据自己的 `max_tokens`。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--num-requests` | `2500` | 请求总数 |
| `--concurrency` | `32` | 仅影响默认文件名和生成的 `run_perf.py` 中的 `parallel` |
| `--cache-hit-rate` | `0.25` | 目标共享前缀比例，范围 `[0, 1)` |
| `--gsm8k-path` | 空 | 本地 GSM8K JSONL；不指定则尝试从 Hugging Face 加载 |
| `--tokenizer` | 空 | Hugging Face tokenizer 名称或本地模型目录 |
| `--kv-block-size` | `0` | 大于 0 时将共享前缀对齐到完整 KV block |
| `--output` | 自动生成 | 输出 JSONL 路径 |
| `--format` | `aisbench` | 输出格式：`aisbench` 或 `evalscope` |
| `--body-salt` | 自动生成 | 控制语料重排、共享块标记、正文块标记和语料偏移 |
| `--seed` | `42` | 控制请求顺序，并参与共享语料偏移计算 |

## 检查生成结果

查看请求数量：

```bash
wc -l <generated.jsonl>
```

查看第一条数据：

```bash
head -n 1 <generated.jsonl> | jq .
```

检查输出长度分布：

```bash
jq -r '.max_tokens' <generated.jsonl> \
  | sort -n \
  | uniq -c
```

检查 EvalScope 数据字段：

```bash
head -n 1 <generated.jsonl> \
  | jq '{keys: keys, max_tokens, ignore_eos, stream, role: .messages[0].role}'
```

比较两轮文件：

```bash
shasum -a 256 batch-A.jsonl batch-B.jsonl
```

两次哈希应不同。

## 常见问题

### tokenizer 被当成在线模型仓库

使用完整绝对路径：

```bash
--tokenizer "/Users/gaosir/Downloads/工作/数据生成脚本/dsv3.2/ds-v3.2-w8a8"
```

### 所有请求失败，指标显示 `-1` 或 `-1000`

检查：

1. URL 是否为 `/v1/chat/completions`。
2. 数据是否包含 `messages`。
3. API Key 是否有效。
4. 网关是否返回 429、连接重置或超时。
5. 服务端最大上下文是否能容纳最长请求。

### 出现 `Connection reset by peer`

通常表示网关、代理或模型服务主动断开连接。先降低压力：

```python
number=100
parallel=32
rate=1
```

### 中断后出现 pending task

压测过程中按下 `Ctrl+C` 时，尚未完成的异步请求可能打印：

```text
Task was destroyed but it is pending
```

这不代表生成的数据集损坏。

### 实际缓存命中率与 25% 不一致

可能原因：

- 服务端没有开启 Prefix Caching。
- `--kv-block-size` 与服务端 KV block 大小不同。
- Chat Template 或系统提示增加了固定 token。
- 第一条请求负责写入缓存，不能命中。
- 请求被路由到不共享缓存的不同实例。
- KV Connector 或网关使用了额外缓存隔离规则。

## 注意事项

- 长度是生成器 tokenizer 测得的用户文本长度，服务端 Chat Template 会增加额外 token。
- 最长档为 `84200 input + 14050 output`，不含 Chat Template 时总长度已经达到 `98250 tokens`。
- 服务端最大上下文长度需要大于最长输入、输出和 Chat Template token 的总和。
- `ignore_eos=true` 需要服务端兼容，否则可能被忽略或返回参数错误。
- 不要在脚本或仓库中明文保存 API Key，建议通过环境变量传入。
