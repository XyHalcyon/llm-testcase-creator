# DeepSeek V4 Flash 混合序列数据集生成器（预热 + 正式双文件）

本目录基于 GSM8K 问题生成 DeepSeek V4 Flash 性能压测数据集，采用**双文件模式**：

- **预热文件**：只含各预热桶的前缀请求（长度 < 正式长度），**先跑**它把前缀 KV 写入服务端 prefix cache。
- **正式文件**：20 档普通混合序列 + 预热桶的完整长度请求，**后跑**它——完整长度请求与预热请求 token 级前缀一致，命中预热写入的缓存，只需计算前缀之外的新增部分。

核心特性：

- **混合序列流量**：20 档输入/输出长度，整体平均输入约 `14.9K tokens`、平均目标输出约 `290 tokens`（取自 rr=5/21机 真实流量画像）。
- **25% 共享前缀**（默认）：每条 prompt 前 25% 取自全局共享前缀。
- **块级批次隔离**：共享区/正文区每个 128-token 块带 `[shared-<batch>-<block>]` / `[body-<batch>-<rid>-<block>]` 标记，跨批完整块交集 = 0。
- **预热/正式前缀对**：三桶 `(正式长度, 预热前缀长度)` = `(344158, 292864)` / `(499835, 448512)` / `(728183, 676864)`，正式长请求只需计算 ~51,200 token（400 个 128-block），其余命中预热缓存。

生成器支持两种输出格式：

- `evalscope`：OpenAI Chat Completions 请求格式，并自动生成 EvalScope 自定义数据集插件和两个运行脚本（预热 + 正式）。
- `aisbench`：AISBench `qa` 自定义数据集格式。

## 预热/正式前缀对（核心机制）

### 设计目标

正式测试中的超长请求（如 728183 token 输入）若无缓存命中，单条 prefill 需要数十秒。通过预热文件提前把长请求的**前缀部分**写入服务端 prefix cache，正式测试时长请求只需计算**前缀之外的新增量**（每桶统一 51,200 token = 400 个 128-block），TTFT 大幅下降。

### 前缀对配置

修改 `gen_data_4k_12k.py` 顶部的 `WARMUP_PAIRS` 常量，增删 entry 即可：

```python
WARMUP_PAIRS = [
    (344158, 292864),   # 正式 344064 tok, 预热 292864 tok, 需计算 51200
    (499835, 448512),   # 正式 499712 tok, 预热 448512 tok, 需计算 51200
    (728183, 676864),   # 正式 728064 tok, 预热 676864 tok, 需计算 51200
]
```

- 每条 entry：`(正式输入长度, 预热前缀长度)`
- 约束：**只要求预热前缀长度 < 正式输入长度**；两个长度会自动 floor 对齐到 `--kv-block-size`（保证整块命中，无部分命中损耗）；正式输入长度必须存在于 `DISTRIBUTION`。
- 也可通过 CLI 覆盖：`--warmup-pairs "344158:292864,499835:448512,728183:676864"`（格式 `正式:预热`）。

### 前缀一致性保证

预热与正式文本由**同一次运行**生成：构造器先拼出完整的正式长度 token 序列

```
full_seq = [shared_prefix(=25%×正式长度, 含批次块标记)] [\n] [pair唯一语料段]
```

然后按两个长度分别截断 decode：

```
预热文本 = decode(full_seq[:预热前缀长度])
正式文本 = decode(full_seq[:正式长度])
```

同一 token 序列两次截断 → 预热是正式的严格前缀（构造保证），并通过：

- **字符级断言**（`assert formal[:len(warmup)] == warmup`，失败即中断）
- **token 级验证**（重新分词后校验前缀，捕获 BPE 边界漂移；极罕见，失败打印警告）

两文件同 salt / 同语料偏移，跨文件一致性由构造保证，无静默失配风险。

### 命中比例与需计算量

| 桶 | 正式长度 | 预热前缀 | 命中% | 需计算 | 需计算块 |
|---|---:|---:|---:|---:|---:|
| 344158 | 344,064 | 292,864 | 85.1% | 51,200 | 400 |
| 499835 | 499,712 | 448,512 | 89.8% | 51,200 | 400 |
| 728183 | 728,064 | 676,864 | 93.0% | 51,200 | 400 |

调整预热前缀长度即可权衡：预热越短 → 预热越快，但正式命中率越低；预热越长 → 命中越高，但预热成本越大。

### 运行前提（重要）

1. **服务端 prefix cache 容量**必须能容纳全部预热 KV（≈ 前缀对组数 × 预热前缀长度 token），否则预热自逐出，正式测试无法命中。
2. **预热与正式必须打到同一个服务实例**（实例重启缓存丢失）。
3. **预热后尽快启动正式测试**，减少缓存被中间流量冲刷的窗口。

## 目录结构

```text
ds-v4-12k/
├── GSM8K.jsonl                 # 基础问题数据，每行包含 question 和 answer
├── gen_data_4k_12k.py          # 数据集生成脚本（双文件：预热 + 正式）
├── evalscope_mixed_plugin.py   # EvalScope 自定义数据集插件
├── gen_4k_12k.sh               # 便捷生成脚本
└── README.md                   # 本说明文件
```

执行生成脚本后，输出目录还会生成：

```text
├── warmup_prefix*.jsonl        # 预热文件
├── gsm8k_4k_12k_*.jsonl        # 正式文件
├── evalscope_mixed_plugin.py   # format=evalscope 时生成或更新
├── run_perf_warmup.py          # 预热 runner（先跑）
└── run_perf.py                 # 正式 runner（后跑）
```

## 长度分布

脚本内置以下输入/输出 token 分布（取自 rr=5/21机 混合序列流量画像，按输入长度升序 20 档），整体平均输入约 `14.9K tokens`、平均目标输出约 `290 tokens`。

| 输入 token | 输出 token | 占比 |
|---:|---:|---:|
| 125 | 74 | 6.09% |
| 622 | 46 | 12.95% |
| 999 | 1,218 | 3.69% |
| 1,643 | 119 | 31.31% |
| 2,624 | 167 | 18.96% |
| 3,147 | 1,045,429 | 0.01% |
| 4,979 | 184 | 9.64% |
| 7,299 | 28,067 | 0.03% |
| 8,958 | 411 | 2.94% |
| 18,997 | 533 | 1.80% |
| 31,130 | 430 | 2.37% |
| 46,871 | 528 | 2.19% |
| 66,476 | 622 | 2.15% |
| 90,801 | 668 | 2.08% |
| 120,954 | 744 | 1.66% |
| 162,151 | 835 | 0.94% |
| 224,544 | 846 | 0.54% |
| 344,158 | 636 | 0.36% |
| 499,835 | 837 | 0.18% |
| 728,183 | 528 | 0.09% |

请求数量通过最大余数法分配到各长度档。三个预热桶（344158/499835/728183）在正式文件中由完整长度请求替代普通请求，并额外生成对应的预热文件请求。

> 尾桶 `3147/1045429` 源报表显示 0.0%（四舍五入），此处按 0.01% 计入：约 1 万条请求出现 1 条超长输出（~1M token）批处理请求；`--num-requests 2160` 时分配 0 条。

## max_tokens 设置规则

数据集中每条请求自带 `max_tokens`，决定该请求的 decode（生成输出）阶段多长。

### 1. 普通请求（17 档）

直接取自 `DISTRIBUTION` 常量中的输出值，输入输出比取决于真实流量画像（非固定比例），如 125→74、1643→119、224544→846 等。

### 2. 预热桶请求

| 文件 | 角色 | max_tokens | 规则 |
|---|---|---:|---|
| 预热文件 | 前缀预热 | **16**（`--warmup-max-tokens`） | 固定极小值，只触发 prefill 写缓存，几乎不 decode |
| 正式文件 | 完整长度请求 | **636 / 837 / 528** | = 该桶 DISTRIBUTION 输出值 |

### 3. 为什么 runner 的 `max_tokens` 必须为 None

`run_perf.py` / `run_perf_warmup.py` 中 `max_tokens` 是全局参数，会覆盖每行数据自带的值。设固定值（如 120000）会使预热请求也生成该长度：

| 维度 | max_tokens=16（设计） | max_tokens=120000（被覆盖） |
|---|---|---|
| 单条预热 decode 耗时 | ~几十 ms | ~几十分钟 |
| prefill 写入的前缀缓存 | 保留 | **被自己的 decode KV 挤出** |
| 正式测试命中 | ✅ | ❌ |

如果确实想用固定输出长度压测，可清空 `WARMUP_PAIRS` 只生成普通混合序列（此时无预热文件）。

## 环境要求与准备

### 1. Python 环境

推荐 Python 3.10 及以上版本。

> 注意：部分环境只安装了 `python3`（没有 `python`）。若使用 uv 管理环境，可用 `uv pip install --system <包名>` 或指定解释器路径（如 `/usr/local/uv/envs/llmcase/bin/python`）。

### 2. 安装依赖

数据生成必需：

```bash
python -m pip install --upgrade pip
python -m pip install transformers tiktoken datasets
```

- `transformers`：加载 DeepSeek V4 Flash tokenizer（`AutoTokenizer.from_pretrained`）
- `tiktoken`：transformers 加载失败时的回退分词器（cl100k_base）
- `datasets`：未指定 `--gsm8k-path` 时从 HuggingFace 拉取 gsm8k

EvalScope 压测需要（仅当实际跑压测时）：

```bash
python -m pip install 'evalscope[perf]==1.9.1'
```

只加载 tokenizer 时出现以下提示不影响数据生成：

```text
PyTorch was not found. Models won't be available and only tokenizers ... can be used.
```

### 3. 下载 DeepSeek V4 Flash Tokenizer

生成器需要 V4 Flash 的 tokenizer 目录控制输入长度精度。**必须用绝对路径**传入 `--tokenizer`，否则会被当成在线仓库名导致加载失败。只需 tokenizer 文件（`tokenizer.json` + `tokenizer_config.json` + `config.json`），约 7.5MB。

**方法一：HuggingFace 镜像（hf-mirror.com，国内推荐）**

```bash
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --include "tokenizer*" "config.json" \
  --local-dir ./deepseek-v4-flash-tokenizer
```

**方法二：ModelScope**

```bash
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-V4-Flash \
  --include "tokenizer*" "config.json" \
  --local_dir ./deepseek-v4-flash-tokenizer
```

下载后验证：

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./deepseek-v4-flash-tokenizer', trust_remote_code=True)
print('词表大小:', tok.vocab_size)
print('编码示例:', tok.encode('Hello, how are you?', add_special_tokens=False)[:10])
"
```

### 4. 数据源

目录已自带 `GSM8K.jsonl`（749KB，每行含 `question` 和 `answer`），通过 `--gsm8k-path "./GSM8K.jsonl"` 指定。

## 快速生成数据集

推荐命令（或直接 `bash gen_4k_12k.sh`）：

```bash
python gen_data_4k_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V4-Flash/" \
  --num-requests 2160 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --min-pairs 3 \
  --warmup-output "./warmup_prefix.jsonl" \
  --format evalscope \
  --rate 4
```

执行后生成：

- **预热文件** `warmup_prefix.jsonl`：15 条前缀请求（组数 = 按占比分配 + `--min-pairs` 提升）
- **正式文件** `gsm8k_4k_12k_c128_cache25_<时间戳>.jsonl`：2160+ 条（20 档普通请求 + 预热桶完整长度请求）
- `run_perf_warmup.py` / `run_perf.py`：两个压测 runner

`--concurrency` 不会改变数据集内容，只会影响默认文件名和 runner 中的 `parallel`。

## 保证两次生成的数据不同

每次不指定 `--body-salt` 时，脚本自动生成唯一盐值（`年月日_时分秒_随机十六进制`），用于：

1. **每批重新排列 GSM8K 语料**：`SHA256(seed:body_salt)` 初始化独立随机顺序。
2. **共享区每个块带批次标记**：`[shared-<batch_tag>-<block>]`，同批一致、跨批不同。
3. **正文区每个块带唯一标记**：`[body-<batch_tag>-<rid>-<block>]`，不同批次/请求/块均不同。
4. **预热/正式前缀文本随批次变化**：语料重排 + 批次标记继承到前缀对文本。
5. **唯一输出文件名**：不指定 `--output` 时自动带时间戳+随机数。

两次默认执行：完整请求文本、文件 SHA-256 均不同；长度档、请求数量、目标 25% 共享比例相同。

注意：若服务端在用户内容前插入超过一个完整 KV block 的固定 Chat Template 或系统提示，这些固定模板块仍可能跨批次命中。需要严格隔离时应使用服务端原生请求级缓存盐值，或在每轮压测前清理 Prefix Cache。

## 输出格式

### EvalScope

每行是一个完整 Chat Completions 请求：

```json
{
  "messages": [{"role": "user", "content": "生成的混合序列文本"}],
  "max_tokens": 1760,
  "ignore_eos": true,
  "stream": true
}
```

预热文件每行 `max_tokens: 16`；正式文件中预热桶完整长度请求的 `max_tokens` 为该桶目标值（636/837/528）。

### AISBench

每行格式：`{"question": "...", "answer": "none", "max_tokens": 1760}`

## 使用 EvalScope 压测

```bash
export OPENAI_API_KEY="你的API_KEY"

# 1. 预热: 把前缀 KV 写入服务端 prefix cache
python run_perf_warmup.py

# 2. 正式: 长请求命中预热前缀
python run_perf.py
```

runner 关键参数（生成时已填好）：

| 参数 | 预热 runner | 正式 runner |
|---|---|---|
| `dataset_path` | warmup_prefix.jsonl | 正式文件 |
| `number` | 预热请求数（如 15） | 正式请求数（如 2161） |
| `parallel` | `--concurrency` | `--concurrency` |
| `rate` | `--rate` | `--rate` |
| `max_tokens` | **None**（必须） | **None**（必须） |

### 关键约束

| 参数 | 要求 | 原因 |
|---|---|---|
| `max_tokens` | **必须 None** | 设固定值会使预热请求也生成该长度，前缀缓存被自己的 decode KV 挤出，正式测试无法命中 |
| 服务实例 | 预热与正式**同一实例** | 实例重启缓存丢失 |
| 时序 | 预热完成后尽快跑正式 | 减少缓存被中间流量冲刷的窗口 |
| 缓存容量 | ≥ 全部预热 KV | 否则预热自逐出 |

建议先小规模验证（`--num-requests 100 --min-pairs 3`），确认成功率后再逐步提高。

## 使用 AISBench 压测

```bash
# 先预热后正式
ais_bench --models <your_model> --custom-dataset-path warmup_prefix.jsonl \
  --custom-dataset-data-type qa --max-out-len -1
ais_bench --models <your_model> --custom-dataset-path <formal.jsonl> \
  --custom-dataset-data-type qa --max-out-len -1
```

`--max-out-len -1` 表示使用数据集中每条请求自己的 `max_tokens`。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--num-requests` | `2500` | 正式文件请求总数 |
| `--concurrency` | `32` | 仅影响默认文件名和 runner 的 `parallel` |
| `--cache-hit-rate` | `0.25` | 共享前缀占每条 prompt 的比例，范围 `[0, 1)` |
| `--gsm8k-path` | 空 | 本地 GSM8K JSONL；不指定则尝试从 Hugging Face 加载 |
| `--tokenizer` | 空 | Hugging Face tokenizer 名称或本地权重目录 |
| `--kv-block-size` | `0` | >0 时把共享前缀和预热/正式长度对齐到完整 KV block |
| `--output` | 自动生成 | 正式文件输出 JSONL 路径 |
| `--warmup-output` | 自动生成 | 预热文件输出 JSONL 路径 |
| `--format` | `aisbench` | 输出格式：`aisbench` 或 `evalscope` |
| `--body-salt` | 自动生成 | 批次唯一值，控制语料重排、共享块标记、正文块标记 |
| `--seed` | `42` | 控制请求打乱顺序，并参与共享语料偏移计算 |
| `--warmup-pairs` | `344158:292864,...` | 预热/正式前缀对，`正式:预热` 格式逗号分隔 |
| `--min-pairs` | `3` | 每个预热桶的最少前缀对组数，不足时自动提升 |
| `--warmup-max-tokens` | `16` | 预热请求输出 tokens（decode 长度） |
| `--rate` | `4.0` | EvalScope rate 参数（写入两个 runner） |

## 生成结果检查

```bash
# 预热文件: 应全部 max_tokens=16
jq -r '.max_tokens' warmup_prefix.jsonl | sort -n | uniq -c

# 正式文件: 应有 636/837/528 三档 target, 无 16 (无 priming)
jq -r '.max_tokens' <formal.jsonl> | sort -n | uniq -c

# 跨文件前缀验证: 每条预热请求应是正式文件中某条长请求的前缀
python3 -c "
import json
w = [json.loads(l) for l in open('warmup_prefix.jsonl') if l.strip()]
f = [json.loads(l) for l in open('<formal.jsonl>') if l.strip()]
ok = sum(1 for wr in w if any(
    len(fr['messages'][0]['content']) >= len(wr['messages'][0]['content'])
    and fr['messages'][0]['content'][:len(wr['messages'][0]['content'])] == wr['messages'][0]['content']
    for fr in f))
print(f'前缀匹配 {ok}/{len(w)}')
"
```

对比两批数据文件是否不同：`shasum -a 256 batch-A.jsonl batch-B.jsonl`

## 常见问题

### 正式测试长请求 TTFT 没有下降

可能原因：

- 预热 runner 没先跑，或跑在不同服务实例上。
- 服务端未开启 Prefix Caching。
- 服务端缓存容量不足，预热 KV 被逐出（检查容量 ≥ 预热 KV 总量）。
- 预热与正式之间隔了太久，中间流量冲刷掉缓存。
- `max_tokens` 设了固定值（必须 None），预热 decode 把前缀 KV 挤出。

### 实际缓存命中率与 25% 不一致

- `--kv-block-size` 与服务端 KV block 大小不一致。
- Chat Template 或系统提示增加了额外固定 token。
- 第一条请求需要写入缓存，本身不能命中。
- 请求被路由到没有共享缓存的实例。

### tokenizer 被当成在线模型仓库

使用完整绝对路径：`--tokenizer "/apps/models/DeepSeek-V4-Flash"`

### 出现 `Connection reset by peer`

降低压力：`number=100, parallel=32, rate=1`，确认成功后逐步提高。

## 注意事项

- 长度为脚本 tokenizer 测得的用户文本长度，服务端 Chat Template 可能增加额外输入 token。
- 最长档为 `728183 input + 528 output`，服务端最大上下文需覆盖输入、输出和 Chat Template token 总和。
- `ignore_eos=true` 需要服务端兼容。
- 预热请求（16 tokens 输出）会略微改变整体输出 token 分布，但影响 <1%。
- 不要把 API Key 明文提交到脚本或代码仓库，推荐通过环境变量传入。
- 预热/正式双文件模式是**缓存命中基准测试**，不是对话保真度测试。真实 agent 对话有助手响应、工具调用、角色标记等额外 token，本方案未模拟这些。
