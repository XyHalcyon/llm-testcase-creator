# DeepSeek-V3.2 3K 混合序列数据集生成器

本目录用于基于 GSM8K 问题生成 DeepSeek-V3.2 性能压测数据集。生成的数据包含多档输入/输出长度、可配置的共享前缀比例，以及用于隔离不同压测批次缓存的唯一标记。

脚本支持两种输出格式：

- `evalscope`：OpenAI Chat Completions 请求格式，并自动生成 EvalScope 自定义数据集插件和运行脚本。
- `aisbench`：AISBench `qa` 自定义数据集格式。

## 目录结构

```text
ds-v3.2-3k/
├── GSM8K.jsonl                 # 基础问题数据，每行包含 question 和 answer
├── gen_data_3k_unique.py       # 数据集生成脚本
├── evalscope_mixed_plugin.py   # EvalScope 自定义数据集插件
└── README.md                   # 本说明文件
```

执行生成脚本后，输出目录还会生成：

```text
├── gsm8k_3k_*.jsonl            # 生成的压测数据集
├── evalscope_mixed_plugin.py   # format=evalscope 时生成或更新
└── run_perf.py                 # EvalScope 压测启动模板
```

## 数据分布

脚本内置以下输入/输出 token 分布，整体平均输入约 `3038 tokens`，平均目标输出约 `434 tokens`。

| 输入 token | 输出 token | 占比 |
|---:|---:|---:|
| 112 | 16 | 2.53% |
| 336 | 48 | 1.41% |
| 672 | 96 | 37.93% |
| 1,313 | 188 | 14.01% |
| 2,625 | 375 | 22.05% |
| 5,250 | 750 | 14.01% |
| 10,500 | 1,500 | 5.08% |
| 21,000 | 3,000 | 2.49% |
| 42,000 | 6,000 | 0.46% |
| 70,000 | 10,000 | 0.04% |

请求数量通过最大余数法分配到各长度档，最终条数严格等于 `--num-requests`。

例如 `--num-requests 7360` 时，各档数量依次为：

```text
186, 104, 2791, 1031, 1623, 1031, 374, 183, 34, 3
```

## 环境要求

推荐使用 Python 3.10 及以上版本和模型自身的 tokenizer。

创建虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install transformers tiktoken datasets
```

使用 EvalScope 压测时安装：

```bash
python -m pip install 'evalscope[perf]==1.9.1'
```

只使用 tokenizer 时出现以下提示不影响数据生成：

```text
PyTorch was not found. Models won't be available and only tokenizers ... can be used.
```

## 快速生成 EvalScope 数据集

进入本目录：

```bash
cd "/Users/gaosir/Downloads/工作/数据生成脚本/dsv3.2/3k/unique_dataset/ds-v3.2-3k"
```

推荐命令：

```bash
python gen_data_3k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../../ds-v3.2-w8a8" \
  --num-requests 7360 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --concurrency 128 \
  --format evalscope
```

不指定 `--output` 时，脚本会使用时间和随机值生成唯一文件名，例如：

```text
gsm8k_3k_c128_cache15_20260722_120000_a1b2c3d4.jsonl
```

同时会在数据集所在目录生成：

```text
evalscope_mixed_plugin.py
run_perf.py
```

`--concurrency` 不会改变数据集内容，只会影响默认文件名和生成的 `run_perf.py` 中的 `parallel`。

## 指定输出文件

```bash
python gen_data_3k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../../ds-v3.2-w8a8" \
  --num-requests 7360 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --format evalscope \
  --output "./mix_3k_unique.jsonl"
```

再次使用相同 `--output` 会覆盖旧文件。需要保留多轮数据时，建议省略 `--output`，或者为每轮使用不同文件名。

## 每次生成不同批次

`--body-salt` 留空时，脚本自动生成如下唯一值：

```text
年月日_时分秒_随机十六进制
```

该盐值用于：

1. 在共享前缀最前面加入同批次公共标记 `[batch-<salt>]`。
2. 根据 `SHA256(seed:salt)` 改变 GSM8K 共享语料的起始偏移。
3. 在共享前缀后加入逐请求标记 `[req-<salt><request_id>]`。
4. 在未指定 `--output` 时生成唯一文件名。

效果如下：

- 同一批次的请求共享批次标记和目标比例的公共前缀。
- 达到共享前缀边界后，逐请求标记使不同请求立即分叉。
- 不同批次具有不同批次标记、共享语料偏移和请求标记。
- 输入/输出长度分布和共享前缀总长度不变。

使用 `--kv-block-size 128` 时，共享前缀会对齐到完整 KV block。对于标准连续前缀缓存，一旦逐请求标记所在块不同，后续即使使用相同 GSM8K 语料，也不会作为连续前缀继续命中。

注意：批次标记位于用户消息内容最前面。如果服务端在用户内容前额外插入超过一个 KV block 的固定 Chat Template 或系统提示，这些固定块仍可能跨批次命中。要求严格隔离时，应同时使用服务端支持的请求级缓存盐值，或者在每轮测试前清理服务端 Prefix Cache。

### 手动指定批次盐值

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_${RANDOM}_${RANDOM}"

python gen_data_3k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../../ds-v3.2-w8a8" \
  --num-requests 7360 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --format evalscope \
  --body-salt "${RUN_ID}" \
  --seed 42 \
  --output "./mix_3k_${RUN_ID}.jsonl"
```

固定全部参数、`--body-salt` 和 `--seed` 可以复现相同数据；改变 `--body-salt` 可生成不同批次。

## 输出格式

### EvalScope

每行是一个完整 Chat Completions 请求：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "生成的混合长度文本"
    }
  ],
  "max_tokens": 375,
  "ignore_eos": true,
  "stream": true
}
```

每条请求携带自己的 `max_tokens`，因此 `run_perf.py` 中必须保持：

```python
max_tokens=None
```

否则 EvalScope 的全局输出长度可能覆盖数据集中的逐请求输出长度。

### AISBench

每行格式：

```json
{
  "question": "生成的混合长度文本",
  "answer": "none",
  "max_tokens": 375
}
```

生成命令：

```bash
python gen_data_3k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "../../../ds-v3.2-w8a8" \
  --num-requests 7360 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --format aisbench
```

## 使用 EvalScope 压测

生成数据后，修改自动生成的 `run_perf.py`：

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

设置 API Key 并运行：

```bash
export OPENAI_API_KEY="你的API_KEY"
python run_perf.py
```

建议先使用小规模请求验证：

```python
number=100
parallel=32
rate=1
```

确认成功率和请求格式正常后，再逐步提高 `number`、`parallel` 和 `rate`。

参数含义：

- `number`：本轮发送的请求总数。
- `parallel`：最多允许的并发或在途请求数。
- `rate=-1`：闭环并发模式，请求完成后继续补发。
- `rate=N`：按约 `N req/s` 的速率发送请求，`parallel` 作为在途请求上限。
- `max_tokens=None`：保留数据集中每条请求自己的输出长度。

## 使用 AISBench 压测

```bash
ais_bench --models <your_model> \
  --custom-dataset-path <generated.jsonl> \
  --custom-dataset-data-type qa \
  --max-out-len -1
```

`--max-out-len -1` 表示使用数据集中每条请求自己的 `max_tokens`。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--num-requests` | `2500` | 生成的请求总数 |
| `--concurrency` | `32` | 仅影响默认文件名和生成的 EvalScope `parallel` |
| `--cache-hit-rate` | `0.15` | 每条请求中计划共享的前缀比例，范围 `[0, 1)` |
| `--gsm8k-path` | 空 | 本地 GSM8K JSONL；不指定则尝试从 Hugging Face 加载 |
| `--tokenizer` | 空 | Hugging Face tokenizer 名称或本地权重目录 |
| `--kv-block-size` | `0` | 大于 0 时将共享前缀对齐到完整 KV block |
| `--output` | 自动生成 | 输出 JSONL 文件路径 |
| `--format` | `aisbench` | 输出格式：`aisbench` 或 `evalscope` |
| `--body-salt` | 自动生成 | 批次唯一值，控制批次标记、请求标记和共享语料偏移 |
| `--seed` | `42` | 控制请求打乱顺序，并参与共享语料偏移计算 |

## 生成结果检查

查看文件数量和第一条数据：

```bash
wc -l <generated.jsonl>
head -n 1 <generated.jsonl> | jq .
```

查看输出长度分布：

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

对比两批数据文件是否不同：

```bash
shasum -a 256 batch-A.jsonl batch-B.jsonl
```

## 常见问题

### tokenizer 被当成 ModelScope 仓库名称

请使用完整绝对路径，并确保开头包含 `/`：

```bash
--tokenizer "/Users/gaosir/Downloads/工作/数据生成脚本/dsv3.2/ds-v3.2-w8a8"
```

不要漏掉最前面的 `/Users/...`。

### 所有请求都失败，指标为 `-1` 或 `-1000`

检查以下内容：

1. URL 是否为 `/v1/chat/completions`。
2. 数据是否使用 `messages`，而不是只包含 `prompt`。
3. API Key 是否正确。
4. 网关是否返回 429、连接重置或超时。

### 出现 `Connection reset by peer`

通常表示网关、代理或模型服务主动断开连接。先降低压力：

```python
number=100
parallel=32
rate=1
```

再逐步增加请求速率。

### 中断后出现 `Task was destroyed but it is pending`

通常是压测过程中按下 `Ctrl+C`，异步请求尚未完成导致的退出提示，不代表数据集损坏。重新运行时使用新的输出目录即可。

### 实际缓存命中率不等于 15%

可能原因包括：

- `--kv-block-size` 与服务端 KV block 大小不一致。
- Chat Template 或系统提示增加了额外固定 token。
- 第一条请求需要写入缓存，本身不能命中。
- 服务端未开启 Prefix Caching。
- 不同请求被路由到没有共享缓存的实例。
- 网关或 KV Connector 使用了不同的缓存隔离规则。

## 注意事项

- 长度为脚本 tokenizer 测得的用户文本长度，服务端 Chat Template 可能增加额外输入 token。
- `ignore_eos=true` 需要服务端兼容；不兼容时可能被忽略或返回参数错误。
- 最长档为 `70000 input + 10000 output`，服务端最大上下文长度至少需要覆盖输入、输出和 Chat Template token。
- 不要把 API Key 明文提交到脚本或代码仓库，推荐通过环境变量传入。
