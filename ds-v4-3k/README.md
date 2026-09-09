# DeepSeek V4 Flash 3K 混合序列数据集生成器（含渐进式前缀链）

本目录用于基于 GSM8K 问题生成 DeepSeek V4 Flash 性能压测数据集。在 ds-v3.2-3k 架构基础上，**核心新增"渐进式前缀链"（Progressive Prefix Chain）**，模拟 agent 多轮对话上下文累积增长场景，使长序列请求能命中前阶产生的 prefix cache，从而降低 TTFT。

生成的数据包含多档输入/输出长度、可配置的共享前缀比例（默认 15%）、用于隔离不同压测批次缓存的唯一标记，以及 70k 长度桶的渐进式前缀链。

脚本支持两种输出格式：

- `evalscope`：OpenAI Chat Completions 请求格式，并自动生成 EvalScope 自定义数据集插件和运行脚本。
- `aisbench`：AISBench `qa` 自定义数据集格式。

## 核心创新：渐进式前缀链

### 设计动机

在 agent 多轮对话场景中，上下文随对话轮次累积增长：第 1 轮 10k tokens，第 2 轮 20k tokens（前 10k 与第 1 轮相同），... 第 7 轮 70k tokens（前 60k 与第 6 轮相同）。如果服务端开启 prefix cache，后阶请求应能命中前阶请求写入的 KV 缓存，显著降低 TTFT。

### 链结构

70k 长度桶的每个请求被替换为 7 阶渐进链：

```
Stage 0: 10k input  → max_tokens=16    (priming, 触发 prefill 写缓存)
Stage 1: 20k input  → max_tokens=16    (前 10k = Stage 0 全部 + 10k 新内容)
Stage 2: 30k input  → max_tokens=16    (前 20k = Stage 1 全部 + 10k 新内容)
Stage 3: 40k input  → max_tokens=16    (前 30k = Stage 2 全部 + 10k 新内容)
Stage 4: 50k input  → max_tokens=16    (前 40k = Stage 3 全部 + 10k 新内容)
Stage 5: 60k input  → max_tokens=16    (前 50k = Stage 4 全部 + 10k 新内容)
Stage 6: 70k input  → max_tokens=10000 (前 60k = Stage 5 全部 + 10k 新内容 ← 目标请求)
```

每阶内容是下一阶的**严格 token 级前缀**。中间阶段（Stage 0-5）仅生成 16 tokens 输出（priming），目标阶段（Stage 6）输出 10000 tokens。

### 内容构造

```
完整 70k token 序列 = [batch_marker] [shared_prefix(固定=15%×70k)] [\n] [chain_unique_segments]
```

- **shared_prefix 长度固定**为 15% × 70k = 10496 tokens（block 对齐后），所有阶段共用。这是前缀链生效的关键——如果按每阶段长度分别计算 shared_len，前缀链会断裂。
- **不插入 `[req-<salt><rid>]` 标记**：该标记在普通请求中用于打断缓存，在链内会破坏前缀连续性。
- **\n 分隔符**：在 shared/segment 边界插入换行符，稳定分词边界。
- **链间唯一性**：每条链用 `chain_idx × _ROTATE_PRIME` 计算不同语料偏移，保证不同链的 segment 内容不同。

### 前缀验证

构造后自动验证 `decode→encode` 前缀一致性（BPE 边界漂移检查）：重新分词每阶段文本，断言 Stage i 的 token 序列是 Stage i+1 的严格前缀。如发生漂移（极罕见），打印警告。

### 交错排列

测试脚本并发发送请求，如果 Stage 6（70k）和 Stage 0（10k）同时到达，Stage 6 无法命中 Stage 0 的缓存（Stage 0 尚未完成 prefill）。解决方案：**链内 stage 间插入 gap 个普通请求作为间隔**。

```
[gap×普通请求], A_s0, B_s0, C_s0, [gap×普通请求], A_s1, B_s1, C_s1, ..., A_s6, B_s6, C_s6, [剩余普通请求]
```

- `gap` 默认 = `--concurrency`，确保 Stage i 的 prefill 完成后 Stage i+1 才到达。
- 同一 stage 的多条链请求紧挨发送（它们互相独立，不依赖彼此缓存）。
- **交错排列后禁止 shuffle**：stage 顺序是缓存命中机制的核心。

## 目录结构

```text
ds-v4-3k/
├── GSM8K.jsonl                 # 基础问题数据，每行包含 question 和 answer
├── gen_data_4k_unique.py       # 数据集生成脚本（含渐进式前缀链）
├── evalscope_mixed_plugin.py   # EvalScope 自定义数据集插件
├── gen_4k.sh                   # 便捷生成脚本
├── run_perf.py                 # EvalScope 压测启动模板
├── run_perf1.py                # EvalScope 压测配置1（256并发, rate=4）
├── run_perf2.py                # EvalScope 压测配置2（287并发, rate=2.1）
└── README.md                   # 本说明文件
```

执行生成脚本后，输出目录还会生成：

```text
├── gsm8k_4k_*.jsonl            # 生成的压测数据集
├── evalscope_mixed_plugin.py   # format=evalscope 时生成或更新
└── run_perf.py                 # EvalScope 压测启动模板（带实际参数）
```

## 数据分布

脚本内置以下输入/输出 token 分布（与 ds-v3.2-3k 一致），整体平均输入约 `3038 tokens`，平均目标输出约 `434 tokens`。

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

请求数量通过最大余数法分配到各长度档，最终条数严格等于 `--num-requests`。70k 桶的请求被替换为渐进链，链请求数 = 链数 × 7（含 priming），为额外请求。

## 环境要求

推荐使用 Python 3.10 及以上版本和模型自身的 tokenizer。

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

推荐命令：

```bash
python gen_data_4k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V4-Flash/" \
  --num-requests 2870 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --concurrency 287 \
  --min-chains 3 \
  --chain-gap 0 \
  --format evalscope \
  --rate 4
```

或直接使用便捷脚本：

```bash
bash gen_4k.sh
```

不指定 `--output` 时，脚本会使用时间和随机值生成唯一文件名，例如：

```text
gsm8k_4k_c287_cache15_20260909_120000_a1b2c3d4.jsonl
```

同时会在数据集所在目录生成：

```text
evalscope_mixed_plugin.py
run_perf.py
```

## 每次生成不同批次

`--body-salt` 留空时，脚本自动生成唯一盐值（`年月日_时分秒_随机十六进制`），用于：

1. 在共享前缀最前面加入同批次公共标记 `[batch-<salt>]`。
2. 根据 `SHA256(seed:salt)` 改变 GSM8K 共享语料的起始偏移。
3. 在共享前缀后加入逐请求标记 `[req-<salt><request_id>]`（仅普通请求，链阶段不含此标记）。
4. 在未指定 `--output` 时生成唯一文件名。

效果：

- 同一批次的请求共享批次标记和目标比例的公共前缀。
- 达到共享前缀边界后，逐请求标记使不同请求立即分叉。
- 不同批次具有不同批次标记、共享语料偏移和请求标记。
- 渐进链的 segment 内容也随批次变化（不同语料偏移）。
- 输入/输出长度分布和共享前缀总长度不变。

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

渐进链阶段的请求格式相同，但 `max_tokens` 不同：

- 中间阶段（priming）：`"max_tokens": 16`
- 目标阶段：`"max_tokens": 10000`

每条请求携带自己的 `max_tokens`，因此 `run_perf.py` 中必须保持：

```python
max_tokens=None
```

否则 EvalScope 的全局输出长度可能覆盖数据集中的逐请求输出长度。特别是**不能设为 120000**，否则 priming 阶段也会生成 120k tokens，完全失去 priming 意义。

### AISBench

每行格式：

```json
{
  "question": "生成的混合长度文本",
  "answer": "none",
  "max_tokens": 375
}
```

## 使用 EvalScope 压测

生成数据后，修改自动生成的 `run_perf.py`：

```python
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="deepseek-v4-flash",
    api="openai",
    url="http://your-gateway:8099/v1/chat/completions",
    dataset="mixed",
    dataset_path="/absolute/path/to/generated.jsonl",
    tokenizer_path="/absolute/path/to/DeepSeek-V4-Flash",
    number=2892,        # 必须包含 priming 请求数
    parallel=287,
    rate=4,             # 渐进链模式: 必须 >0, 不能用 -1
    max_tokens=None,    # 别改: None 才逐请求生效
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
```

设置 API Key 并运行：

```bash
export OPENAI_API_KEY="你的API_KEY"
python run_perf.py
```

### 渐进链模式的关键约束

| 参数 | 要求 | 原因 |
|---|---|---|
| `rate` | **必须 > 0** | rate=-1（闭环并发）使 stage 时序不可控，后阶可能在前阶 prefill 完成前到达 |
| `max_tokens` | **必须 None** | 设固定值会使 priming 阶段也生成该长度，失去 priming 意义 |
| `number` | **必须含 priming** | 减少 number 会截断链阶段，破坏前缀链 |
| `parallel` | ≥ `--chain-gap` | 确保间隔请求能填满并发窗口 |

建议先使用小规模请求验证：

```python
number=100
parallel=32
rate=1
```

确认成功率和请求格式正常后，再逐步提高 `number`、`parallel` 和 `rate`。

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
| `--num-requests` | `2500` | 生成的请求总数（不含 priming） |
| `--concurrency` | `32` | 仅影响默认文件名和生成的 EvalScope `parallel` |
| `--cache-hit-rate` | `0.15` | 每条请求中计划共享的前缀比例，范围 `[0, 1)` |
| `--gsm8k-path` | 空 | 本地 GSM8K JSONL；不指定则尝试从 Hugging Face 加载 |
| `--tokenizer` | 空 | Hugging Face tokenizer 名称或本地权重目录 |
| `--kv-block-size` | `0` | 大于 0 时将共享前缀和链阶段对齐到完整 KV block |
| `--output` | 自动生成 | 输出 JSONL 文件路径 |
| `--format` | `aisbench` | 输出格式：`aisbench` 或 `evalscope` |
| `--body-salt` | 自动生成 | 批次唯一值，控制批次标记、请求标记和共享语料偏移 |
| `--seed` | `42` | 控制请求打乱顺序，并参与共享语料偏移计算 |
| `--chain-stages` | `10000,20000,...,70000` | 渐进链阶段输入长度 |
| `--chain-gap` | `0` (auto) | 链 stage 间间隔的普通请求数；0=自动取 concurrency |
| `--chain-intermediate-output` | `16` | 链中间阶段输出 tokens（priming） |
| `--min-chains` | `3` | 70k 桶最少链数，不足时自动提升 |
| `--no-chain` | `False` | 禁用渐进式前缀链，降级为 3k 行为 |
| `--rate` | `4.0` | EvalScope rate 参数（生成 run_perf.py 用；渐进链模式必须 >0） |

## 渐进链工作原理详解

### 1. 名额分配与链数提升

`--num-requests 2870` 时，70k 桶按 0.04% 分配仅得到 1 条请求 = 1 条链。`--min-chains 3`（默认）会自动提升到 3 条链，保证统计意义。

```
70k 桶原始分配 = 1 → 提升到 3 (--min-chains)
总请求 = 2870 普通请求 + 3 链 × 7 阶段 = 2870 + 21 = 2891
其中 priming 请求 = 3 × 6 = 18
```

### 2. 70k token 序列构造

```
full_seq = [batch_marker (~5 tok)] [shared_prefix (10496 tok)] [\n (1 tok)] [chain_segments (~59498 tok)]
                                        ↑ 固定长度 = 15% × 70000, block 对齐
总计 = 70000 tokens
```

### 3. 阶段截断

```
Stage 0 = decode(full_seq[: 9984])     # block 对齐: floor(10000/128)×128
Stage 1 = decode(full_seq[: 19968])    # = 2 × 9984
Stage 2 = decode(full_seq[: 29952])    # = 3 × 9984
...
Stage 6 = decode(full_seq[: 69888])    # ≈ 70000
```

### 4. 前缀验证

```python
for i in range(6):
    stage_i_tokens = tok.encode(stage_texts[i])
    stage_next_tokens = tok.encode(stage_texts[i+1])
    assert stage_next_tokens[:len(stage_i_tokens)] == stage_i_tokens
```

### 5. 交错排列

```
[gap×normal], Chain_A_s0, Chain_B_s0, Chain_C_s0,    # 3条链的Stage 0
[gap×normal], Chain_A_s1, Chain_B_s1, Chain_C_s1,    # 3条链的Stage 1
...
[gap×normal], Chain_A_s6, Chain_B_s6, Chain_C_s6,    # 3条链的Stage 6 (目标)
[remaining normal]
```

`gap = concurrency`（默认），确保 Stage i 的 prefill 完成（写缓存）后 Stage i+1 才到达。

### 6. Stage 0 特殊情况

当 `stage_len (9984) < shared_len (10496)` 时，Stage 0 几乎完全由共享前缀组成。这是**可接受的行为**——Stage 0 仅预热共享前缀缓存，Stage 1+ 才添加唯一 segment 内容。由于普通请求也共享同一前缀，Stage 0 的缓存可能已被普通请求预热。

## 与 ds-v3.2-3k 的区别

| 维度 | ds-v3.2-3k | ds-v4-3k |
|---|---|---|
| 目标模型 | DeepSeek V3.2 | DeepSeek V4 Flash |
| 架构 | 混合序列 + 15% 共享前缀 | 相同 + 渐进式前缀链 |
| 70k 桶处理 | 单一 70k 请求 | 7 阶渐进链 (10k→70k) |
| 缓存命中机制 | 仅 15% 共享前缀 | 15% 共享前缀 + 链内 100% 前缀命中 |
| priming 请求 | 无 | 每链 6 个 (16 tokens 输出) |
| 交错排列 | 无 (全部 shuffle) | 链 stage 间 gap 间隔 |
| run_perf rate | 可用 -1 | **必须 > 0** |
| run_perf max_tokens | None (建议) | **必须 None** |
| run_perf number | = num-requests | **= num-requests + priming** |
| 新增参数 | — | `--chain-stages`, `--chain-gap`, `--chain-intermediate-output`, `--min-chains`, `--no-chain`, `--rate` |

## 降级模式

使用 `--no-chain` 可禁用渐进式前缀链，降级为与 ds-v3.2-3k 完全一致的行为：

```bash
python gen_data_4k_unique.py --no-chain --format evalscope
```

此时 70k 桶使用普通请求（含 `[req-<salt><rid>]` 标记），无 priming，无交错排列，`run_perf.py` 可使用 `rate=-1`。

## 生成结果检查

查看文件数量和第一条数据：

```bash
wc -l <generated.jsonl>
head -n 1 <generated.jsonl> | jq .
```

查看输出长度分布：

```bash
jq -r '.max_tokens' <generated.jsonl> | sort -n | uniq -c
```

检查渐进链阶段（应看到 `max_tokens: 16` 的 priming 请求）：

```bash
jq -r '.max_tokens' <generated.jsonl> | sort -n | uniq -c
# 应包含:
#    18  16       ← 3链 × 6 priming
#    1   10000    ← 3链 × 1 target (70k桶原1+提升2=3)
```

对比两批数据文件是否不同：

```bash
shasum -a 256 batch-A.jsonl batch-B.jsonl
```

## 常见问题

### 渐进链 priming 请求生成了 120k tokens

检查 `run_perf.py` 中的 `max_tokens`。必须为 `None`，不能是 `120000` 或其他固定值。`None` 让数据集中每条请求自带的 `max_tokens` 生效（priming=16, target=10000）。

### 渐进链缓存命中率低

可能原因：

- `rate=-1`（闭环并发）：stage 时序不可控，后阶在前阶 prefill 完成前到达。改为 `rate=N`（N>0）。
- `--chain-gap` 过小：stage 间隔不足，前阶 prefill 未完成。保持 `--chain-gap=0`（auto=concurrency）或增大。
- `--kv-block-size` 与服务端不一致：阶段边界不对齐，缓存部分命中。确认服务端 KV block 大小。
- 服务端未开启 Prefix Caching。
- 服务端缓存不足，早期 stage 被驱逐。
- 不同链请求被路由到不共享缓存的不同实例。

### tokenizer 被当成 ModelScope 仓库名称

请使用完整绝对路径，并确保开头包含 `/`：

```bash
--tokenizer "/apps/models/DeepSeek-V4-Flash"
```

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

## 注意事项

- 长度为脚本 tokenizer 测得的用户文本长度，服务端 Chat Template 可能增加额外输入 token。
- `ignore_eos=true` 需要服务端兼容；不兼容时可能被忽略或返回参数错误。
- 最长档为 `70000 input + 10000 output`，服务端最大上下文长度至少需要覆盖输入、输出和 Chat Template token。
- 渐进链的 priming 请求（16 tokens 输出）会略微改变整体输出 token 分布，但影响 <1%。
- 不要把 API Key 明文提交到脚本或代码仓库，推荐通过环境变量传入。
- 渐进链模式是**缓存命中基准测试**，不是对话保真度测试。真实 agent 对话有助手响应、工具调用、角色标记等额外 token，本方案未模拟这些。
