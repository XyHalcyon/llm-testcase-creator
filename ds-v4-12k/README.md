# DeepSeek V4 Flash 12K 混合序列数据集生成器（含渐进式前缀链）

本目录融合 **ds-v3.2-12k**（12K 重流量模型 + 块级批次隔离）与 **ds-v4-3k**（渐进式前缀链）两套架构，基于 GSM8K 问题生成 DeepSeek V4 Flash 性能压测数据集。

- **12K 流量**：10 档输入/输出长度，整体平均输入约 `12349 tokens`、平均目标输出约 `2056 tokens`。
- **25% 共享前缀**（默认）：每条 prompt 前 25% 取自全局共享前缀。
- **块级批次隔离**：共享区/正文区每个 128-token 块带 `[shared-<batch>-<block>]` / `[body-<batch>-<rid>-<block>]` 标记，跨批完整块交集 = 0。
- **渐进式前缀链**（42100 + 84200 双桶）：将 42100 和 84200 两个长输入桶的请求替换为 7 阶渐进链（如 12k→24k→…→84200），模拟 agent 多轮对话上下文累积增长，使长序列请求命中前阶 prefix cache，降低 TTFT。哪些桶做链可在 `CHAIN_BUCKETS` 常量或 `--chain-buckets` 参数中简单配置。

生成器支持两种输出格式：

- `evalscope`：OpenAI Chat Completions 请求格式，并自动生成 EvalScope 自定义数据集插件和运行脚本。
- `aisbench`：AISBench `qa` 自定义数据集格式。

## 核心创新：渐进式前缀链（多桶）

### 设计动机

在 agent 多轮对话场景中，上下文随对话轮次累积增长：第 1 轮 12k tokens，第 2 轮 24k tokens（前 12k 与第 1 轮相同），… 第 7 轮 84200 tokens（前 72k 与第 6 轮相同）。如果服务端开启 prefix cache，后阶请求应能命中前阶请求写入的 KV 缓存，显著降低 TTFT。

### 哪些桶做链（可配置）

脚本默认对 **42100 和 84200 两个长输入桶**启用渐进链。可通过两种方式配置：

**方式一：改源码常量**（推荐，精细控制每桶阶段长度）

修改 `gen_data_4k_12k.py` 顶部的 `CHAIN_BUCKETS` 常量，增删 entry 即可：

```python
CHAIN_BUCKETS = [
    (42100, [6000, 12000, 18000, 24000, 30000, 36000, 42100]),  # 每轮 ~6k 新内容
    (84200, [12000, 24000, 36000, 48000, 60000, 72000, 84200]), # 每轮 ~12k 新内容
]
```

- 每条 entry：`(bucket_input_len, [stage_lengths])`
- 约束：stages 必须严格递增且最后一阶 == bucket_input_len；所有链桶阶段数必须相同（交错排列约束）。
- 注释掉或删除某行即可禁用该桶的链；添加新桶（须在 DISTRIBUTION 中）即可扩展。

**方式二：CLI 参数**（快速指定哪些桶做链，阶段自动均分）

```bash
# 只对 84200 桶做链 (阶段自动均分 7 阶 + block 对齐)
python gen_data_4k_12k.py --chain-buckets 84200 ...

# 对 42100 和 84200 都做链 (默认; 阶段用 CHAIN_BUCKETS 常量定义)
python gen_data_4k_12k.py --chain-buckets 42100,84200 ...

# 禁用所有链
python gen_data_4k_12k.py --no-chain ...
```

`--chain-buckets` 指定的桶若已在 `CHAIN_BUCKETS` 常量中定义阶段则直接复用；未定义的桶按 `--chain-num-stages`（默认 7）自动均分 + block 对齐生成阶段。

### 链结构

每个链桶的请求被替换为 7 阶渐进链：

```
42100 桶 (每轮 ~6k 新内容):
  Stage 0: 6000 input   → max_tokens=16    (priming)
  Stage 1: 12000 input  → max_tokens=16    (前 6k = Stage 0 + 6k 新内容)
  ...
  Stage 6: 42100 input  → max_tokens=7000  (目标请求, 前 36k = Stage 5 + 6k 新内容)

84200 桶 (每轮 ~12k 新内容):
  Stage 0: 12000 input  → max_tokens=16    (priming)
  Stage 1: 24000 input  → max_tokens=16    (前 12k = Stage 0 + 12k 新内容)
  ...
  Stage 6: 84200 input  → max_tokens=14050 (目标请求, 前 72k = Stage 5 + 12k 新内容)
```

每阶内容是下一阶的**严格 token 级前缀**。中间阶段（Stage 0-5）仅生成 16 tokens 输出（priming），目标阶段（Stage 6）输出该桶的目标输出长度（42100 桶 = 7000，84200 桶 = 14050）。

### 内容构造

```
完整 bucket_len token 序列 = [shared_prefix(固定=25%×bucket_len, 含批次块标记)] [\n] [chain_unique_segments]
```

- **shared_prefix 长度固定**为 25% × 该桶长度（block 对齐后），该桶所有阶段共用。这是前缀链生效的关键。
- **不插入 `[body-<batch>-<rid>-<block>]` 标记**：该标记在普通请求中用于打断缓存，在链内会破坏前缀连续性。
- **\n 分隔符**：在 shared/segment 边界插入换行符，稳定分词边界。
- **链间唯一性**：每条链用 `chain_idx × _ROTATE_PRIME` 计算不同语料偏移，保证不同链的 segment 内容不同。
- **跨桶共享前缀兼容**：42100 桶的 shared_len (10496) < 84200 桶的 shared_len (20992)，二者取自同一段 `shared_ids`，因此 42100 链的 shared 段是 84200 链 shared 段的前缀——42100 链的 priming 会预热 84200 链 shared 段的前半部分，带来额外缓存命中。

### 前缀验证

构造后自动验证 `decode→encode` 前缀一致性（BPE 边界漂移检查）：重新分词每阶段文本，断言 Stage i 的 token 序列是 Stage i+1 的严格前缀。如发生漂移（极罕见），打印警告。

### 缓存命中分析

#### token 级前缀验证

用真实 V4 Flash tokenizer 对一条 84200 链（k=191）做 token 级前缀验证：

```
s0: 11878 tokens
s1: 23883 tokens (前 11878 = s0)  ✓
s2: 35915 tokens (前 23883 = s1)  ✓
s3: 47947 tokens (前 35915 = s2)  ✓
s4: 59851 tokens (前 47947 = s3)  ✓
s5: 71883 tokens (前 59851 = s4)  ✓
s6: 84043 tokens (前 71883 = s5)  ✓
```

所有 6 个 stage 转换的 token 级前缀完全匹配（本批次盐值刚好让截断点落在干净 BPE 边界上）。

#### BPE 边界漂移

不同批次盐值可能导致截断点落在 BPE token 的中间字节，此时 `decode→encode` 非恒等，边界 1-2 个 token 不同（字符级前缀仍成立）。脚本 `_verify_prefix_chain` 会打印警告，影响仅限边界 1 个 KV block（<0.01%），不影响整体命中率。

#### KV block 级命中分析

服务端 KV cache 按 block（如 128 token）粒度管理。以 s0→s1 为例：

| block 范围 | tokens | s0 写入 | s1 命中 | 状态 |
|:---:|---:|---:|---:|:---|
| 0–91 | 0–11775 | 92 个完整 block | 92 个完整 block | ✅ 100% 命中 |
| 92 | 11776–11903 | 前 102/128 token | 前 102 token 匹配 + 26 个新 token | ⚡ 部分命中（80%） |
| 93–186 | 11904–23935 | 未写入 | 93 个全新 block | ❌ 全新计算 |

各 stage 转换的理论命中率：

| 阶段转换 | 总 token | 可命中 token | 理论命中率 |
|:---:|---:|---:|---:|
| s0→s1 | 11,878 + 12,005 | ≈ 11,852 | 98.7% |
| s1→s2 | 23,883 + 12,032 | ≈ 23,808 | 98.9% |
| s2→s3 | 35,915 + 12,032 | ≈ 35,840 | 98.8% |
| s3→s4 | 47,947 + 11,904 | ≈ 47,872 | 98.7% |
| s4→s5 | 59,851 + 12,032 | ≈ 59,776 | 98.7% |
| s5→s6 | 71,883 + 12,160 | ≈ 71,808 | 98.6% |

#### 缓存命中的前提条件

缓存命中保证需要以下 4 个前提同时满足：

| 前提 | 说明 |
|---|---|
| **同一实例 + 前缀缓存开启** | 链内请求必须被路由到同一 batch/实例，且服务端开启了 Prefix Caching（如 vLLM `--enable-prefix-caching`） |
| **缓存未被驱逐** | s0→s1 之间有 gap 个普通请求 + 其他链的 stage 0 作为间隔，若缓存容量不足，s0 写入的 KV 可能被逐出 |
| **请求发送时序受 `rate` 控制** | 必须用 `rate=N>0`（固定速率），不能用 `rate=-1`（闭环并发），否则 stage 到达顺序不可控 |
| **KV block 大小一致** | 生成时 `--kv-block-size 128` 须与服务端实际的 KV block 大小一致 |

### 交错排列

测试脚本并发发送请求，如果 Stage 6（84200）和 Stage 0（12000）同时到达，Stage 6 无法命中 Stage 0 的缓存（Stage 0 尚未完成 prefill）。解决方案：**所有链桶的链统一交错排列，链内 stage 间插入 gap 个普通请求作为间隔**。

```
[gap×普通], 链A_s0, 链B_s0, ..., 链M_s0, [gap×普通], 链A_s1, 链B_s1, ..., 链M_s1, ..., 链A_s6, 链B_s6, ..., 链M_s6, [剩余普通]
```

其中链 A..M 包括 42100 桶的所有链和 84200 桶的所有链（同一 stage 组内混排，互相独立不依赖彼此缓存）。

- `gap` 默认 = `--concurrency`，确保 Stage i 的 prefill 完成后 Stage i+1 才到达。
- 同一 stage 的多条链请求紧挨发送。
- **交错排列后禁止 shuffle**：stage 顺序是缓存命中机制的核心。
- **所有链桶阶段数必须相同**（默认都是 7 阶），否则交错排列报错。

### Stage 0 特殊情况

当 `stage_len (5888) < shared_len (10496)` 时，42100 桶的 Stage 0 几乎完全由共享前缀组成。这是**可接受的行为**——Stage 0 仅预热共享前缀缓存，Stage 1+ 才添加唯一 segment 内容。由于普通请求也共享同一前缀，Stage 0 的缓存可能已被普通请求预热。

## 目录结构

```text
ds-v4-12k/
├── GSM8K.jsonl                 # 基础问题数据，每行包含 question 和 answer
├── gen_data_4k_12k.py          # 数据集生成脚本（含渐进式前缀链 + 块级批次隔离）
├── evalscope_mixed_plugin.py   # EvalScope 自定义数据集插件
├── gen_4k_12k.sh               # 便捷生成脚本
└── README.md                   # 本说明文件
```

执行生成脚本后，输出目录还会生成：

```text
├── gsm8k_4k_12k_*.jsonl        # 生成的压测数据集
├── evalscope_mixed_plugin.py   # format=evalscope 时生成或更新
└── run_perf.py                 # EvalScope 压测启动模板（带实际参数）
```

## 长度分布

脚本内置以下输入/输出 token 分布（与 ds-v3.2-12k 一致），整体平均输入约 `12349 tokens`，平均目标输出约 `2056 tokens`。

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

请求数量通过最大余数法分配到各长度档，最终条数严格等于 `--num-requests`。42100 和 84200 桶的请求被替换为渐进链，链请求数 = 链数 × 7（含 priming），为额外请求。

## max_tokens 设置规则

数据集中每条请求自带 `max_tokens`，决定该请求的 decode（生成输出）阶段多长。设置规则分两类：

### 1. 普通请求（8 档）— 输出 ≈ 输入 ÷ 6

普通请求的 `max_tokens` 按**输出 ≈ 输入 × 1/6**的比例设置，全部硬编码在 `DISTRIBUTION` 常量里：

| 输入 token | 输出 max_tokens | 输出/输入比 |
|---:|---:|---:|
| 55 | 10 | 1:5.5 |
| 330 | 55 | 1:6.0 |
| 660 | 110 | 1:6.0 |
| 1,320 | 220 | 1:6.0 |
| 2,630 | 440 | 1:6.0 |
| 5,266 | 870 | 1:6.0 |
| 10,500 | 1,760 | 1:6.0 |
| 21,060 | 3,500 | 1:6.0 |

模拟真实对话场景"用户输入一定长度的问题，模型回答约输入长度的 1/6"，使平均输出约 2056 token（中等长度结构化回答）。

### 2. 渐进链请求 — 三档固定值

42100 和 84200 桶的请求被替换为渐进链，链内各阶段的 `max_tokens` 由角色决定：

| 阶段 | 角色 | max_tokens | 规则 |
|---|---|---:|---|
| Stage 0–5 | priming（预热缓存） | **16** | 固定极小值，只触发 prefill 写缓存，几乎不 decode |
| 42100 桶 Stage 6 | target（目标请求） | **7000** | = 该桶 DISTRIBUTION 输出值（42100 ÷ 6 ≈ 7000） |
| 84200 桶 Stage 6 | target（目标请求） | **14050** | = 该桶 DISTRIBUTION 输出值（84200 ÷ 6 ≈ 14050） |

- **priming 的 16** 是经验值——足够触发 prefill（模型必须读取全部输入才能生成第 1 个 token），又足够短（16 token decode 只需几十毫秒），不会长时间占用 KV 缓存。
- **target 的 7000/14050** 与该桶普通请求应有的输出长度一致，使渐进链不改变整体输出长度分布（priming 的 16 token 对整体输出分布影响 <1%）。

### 3. 为什么渐进链模式 `max_tokens` 必须为 None

`run_perf.py` 中 `max_tokens` 是全局参数，会覆盖每行数据自带的值。设固定值（如 120000）会覆盖所有 1218 个 priming 请求的 16，使它们也生成 120000 token，**彻底破坏缓存命中机制**：

| 维度 | max_tokens=16（设计） | max_tokens=120000（被覆盖） |
|---|---|---|
| 单个 priming decode 耗时 | ~几十 ms | ~几十分钟 |
| priming decode 产生的 KV | 16 token | 120000 token |
| prefill 写入的前缀缓存 | 保留（decode KV 极小） | **被自己的 decode KV 挤出** |
| Stage i+1 到达时 | 缓存在 → 命中 | 缓存被驱逐 → **无法命中** |
| 1218 个 priming 总输出 | 19,488 token | 1.46 亿 token |
| 全部跑完耗时 | ~3 分钟 | ~283 小时 |

因果链：

```
max_tokens=120000
  → priming 请求 decode 120000 token
  → decode 产生的 120000 个 KV 占满缓存
  → prefill 阶段写入的输入前缀 KV 被 LRU 驱逐
  → Stage i+1 到达时, 前缀缓存已不存在
  → 缓存命中失败 → 渐进式前缀链完全失效
```

如果确实想用固定输出长度压测，应使用 `--no-chain` 禁用渐进链，降级为普通混合序列模式，那时设 `max_tokens=120000` 不会破坏任何机制（因为没有 priming 请求）。

## 环境要求与准备

### 1. Python 环境

推荐 Python 3.10 及以上版本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

> 注意：部分环境只安装了 `python3`（没有 `python`），执行脚本时用 `python3 gen_data_4k_12k.py`。若使用 uv 管理环境，可用 `uv pip install --system <包名>` 直接安装到系统环境。

### 2. 安装依赖

数据生成必需（`gen_data_4k_12k.py` 实际 import）：

```bash
python -m pip install --upgrade pip
python -m pip install transformers tiktoken datasets
```

- `transformers`：加载 DeepSeek V4 Flash tokenizer（`AutoTokenizer.from_pretrained`）
- `tiktoken`：transformers 加载失败时的回退分词器（cl100k_base）
- `datasets`：未指定 `--gsm8k-path` 时从 HuggingFace 拉取 gsm8k

EvalScope 压测需要（仅当 `--format evalscope` 且要实际跑压测时）：

```bash
python -m pip install 'evalscope[perf]==1.9.1'
```

只加载 tokenizer 时出现以下提示不影响数据生成：

```text
PyTorch was not found. Models won't be available and only tokenizers ... can be used.
```

### 3. 下载 DeepSeek V4 Flash Tokenizer

生成器需要 V4 Flash 的 tokenizer 目录控制输入长度精度。**必须用绝对路径**传入 `--tokenizer`，否则会被当成在线仓库名导致加载失败。只需 tokenizer 文件（`tokenizer.json` + `tokenizer_config.json` + `config.json`），约 7.5MB，无需下载几十 GB 权重。

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

**方法三：直连 HuggingFace（需能访问 huggingface.co）**

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --include "tokenizer*" "config.json" \
  --local-dir ./deepseek-v4-flash-tokenizer
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

> 仓库名已核实：HuggingFace 和 ModelScope 均为 `deepseek-ai/DeepSeek-V4-Flash`。用错 tokenizer（如 V3.2 的）会导致实际 token 数与目标分布漂移。

### 4. 数据源

目录已自带 `GSM8K.jsonl`（749KB，每行含 `question` 和 `answer`），通过 `--gsm8k-path "./GSM8K.jsonl"` 指定。不传时尝试从 HuggingFace 在线加载，需网络可达。

## 快速生成 EvalScope 数据集

推荐命令：

```bash
python gen_data_4k_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V4-Flash/" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --min-chains 3 \
  --chain-gap 0 \
  --format evalscope \
  --rate 4
```

或直接使用便捷脚本：

```bash
bash gen_4k_12k.sh
```

不指定 `--output` 时，脚本会使用时间和随机值生成唯一文件名，例如：

```text
gsm8k_4k_12k_c128_cache25_20260909_120000_a1b2c3d4.jsonl
```

同时会在数据集所在目录生成：

```text
evalscope_mixed_plugin.py
run_perf.py
```

`--concurrency` 不会改变数据集内容，只会影响默认文件名和生成的 `run_perf.py` 中的 `parallel`。

## 保证两次生成的数据不同

`gen_data_4k_12k.py` 继承 ds-v3.2-12k 的块级批次隔离逻辑。

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

### 5. 渐进链的 segment 内容也随批次变化

渐进链的 segment 用 `chain_idx × _ROTATE_PRIME` 计算不同语料偏移，且语料本身已按批次重新排列，因此不同批次的链段内容也不同。

### 6. 唯一输出文件名

未指定 `--output` 时，每轮都会生成不同名称的 JSONL 文件，不会覆盖上一轮数据。

这些标记包含在原计划输入 token 长度中，不会额外增加输入长度。两次默认执行具有：

- 不同 GSM8K 问题排列顺序。
- 不同共享语料和正文偏移。
- 不同共享块标记。
- 不同正文块标记。
- 不同渐进链 segment 内容。
- 不同完整请求文本。
- 不同输出文件名和文件 SHA-256。
- 相同长度档、请求数量、输出长度分布和目标 25% 批内共享比例。

注意：标记位于用户消息内容中。如果服务端在用户内容前插入超过一个完整 KV block 的固定 Chat Template 或系统提示，这些固定模板块仍可能跨批次命中。需要严格隔离整个服务端 Prompt 时，应使用服务端原生请求级缓存盐值，或在每轮压测前清理 Prefix Cache。

## 输出格式

### EvalScope

每行是一个完整 Chat Completions 请求：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "生成的 12K 混合序列文本"
    }
  ],
  "max_tokens": 1760,
  "ignore_eos": true,
  "stream": true
}
```

渐进链阶段的请求格式相同，但 `max_tokens` 不同：

- 中间阶段（priming）：`"max_tokens": 16`
- 目标阶段：`"max_tokens": 14050`

每条请求携带自己的 `max_tokens`，因此 `run_perf.py` 中必须保持：

```python
max_tokens=None
```

否则 EvalScope 的全局输出长度可能覆盖数据集中的逐请求输出长度。特别是**不能设为 14050 或更大值**，否则 priming 阶段也会生成该长度，完全失去 priming 意义。

### AISBench

每行格式：

```json
{
  "question": "生成的 12K 混合序列文本",
  "answer": "none",
  "max_tokens": 1760
}
```

## 使用 EvalScope 压测

生成数据后，目录下会有三个 `run_perf` 脚本（参考 ds-v3.2-12k 的三件套结构）：

| 脚本 | 来源 | 用途 |
|---|---|---|
| `run_perf.py` | 生成器自动生成 | 模板，`model/url/tokenizer_path` 是占位符，需改后使用 |
| `run_perf1.py` | 手动创建 | 配置 1：低速率验证（rate=1，先确认服务与请求格式正常） |
| `run_perf2.py` | 手动创建 | 配置 2：中速率压测（rate=4，吞吐更高） |

### run_perf.py（模板，自动生成）

```python
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="<your-served-model-name>",
    api="openai",
    url="http://<HOST>:<PORT>/v1/chat/completions",
    dataset="mixed",
    dataset_path="/absolute/path/to/generated.jsonl",
    tokenizer_path="<tokenizer 或权重目录>",
    number=4878,        # 必须包含 priming 请求数
    parallel=128,
    rate=4,             # 渐进链模式: 必须 >0, 不能用 -1
    max_tokens=None,    # 别改: None 才逐请求生效
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
```

### run_perf1.py（配置 1：低速率验证）

```python
args = Arguments(
    model="deepseek-v4-flash",
    api="openai",
    url="http://localhost:8099/v1/chat/completions",
    dataset="mixed",
    dataset_path='/workspace/llm-testcase-creator/ds-v4-12k/gsm8k_4k_12k_c128_cache25_20260910_072644_53b9b8de.jsonl',
    tokenizer_path="/apps/models/DeepSeek-V4-Flash/",
    number=4878,
    parallel=128,
    rate=1,           # 低速率先验证
    max_tokens=None,  # 别改: None 才逐请求生效 (priming=16, target=7000/14050)
    stream=True,
    name="mix_perf",
)
```

### run_perf2.py（配置 2：中速率压测）

```python
args = Arguments(
    model="deepseek-v4-flash",
    api="openai",
    url="http://localhost:8098/v1/chat/completions",
    dataset="mixed",
    dataset_path='/workspace/llm-testcase-creator/ds-v4-12k/gsm8k_4k_12k_c128_cache25_20260910_072644_53b9b8de.jsonl',
    tokenizer_path="/apps/models/DeepSeek-V4-Flash/",
    number=4878,
    parallel=128,
    rate=4,           # 中速率压测
    max_tokens=None,  # 别改: None 才逐请求生效 (priming=16, target=7000/14050)
    stream=True,
    name="mix_perf",
)
```

### 与 ds-v3.2-12k 的 run_perf 脚本差异

| 项 | ds-v3.2-12k run_perf1/2 | ds-v4-12k run_perf1/2 |
|---|---|---|
| `max_tokens` | `120000`（固定输出长度） | **`None`**（逐请求生效，渐进链必须） |
| `rate` | `1` / `1.56` | `1` / `4` |
| `number` | `1220` / `2620`（= num-requests） | `4878`（= num-requests + priming） |
| `parallel` | `122` / `262` | `128` |
| `model` | `deepseek` | `deepseek-v4-flash` |
| `tokenizer_path` | V3.2 权重目录 | V4 Flash 权重目录 |

> v3.2-12k 的 run_perf1/2 用 `max_tokens=120000` 统一覆盖输出长度，因为它没有渐进链（无 priming）。v4-12k **必须用 `None`**，否则 priming 的 16 会被覆盖成 120000，缓存命中机制失效。

### 运行

```bash
export OPENAI_API_KEY="你的API_KEY"
python3 run_perf1.py   # 低速验证
python3 run_perf2.py   # 中速压测
```

两套配置（rate=1 vs rate=4）用于对比不同到达速率下渐进链的缓存命中率和 TTFT 表现。

### 渐进链模式的关键约束

| 参数 | 要求 | 原因 |
|---|---|---|
| `rate` | **必须 > 0** | rate=-1（闭环并发）使 stage 时序不可控，后阶可能在前阶 prefill 完成前到达 |
| `max_tokens` | **必须 None** | 设固定值会使 priming 阶段也生成该长度，失去 priming 意义（详见上节"为什么渐进链模式 max_tokens 必须为 None"） |
| `number` | **必须含 priming** | 减少 number 会截断链阶段，破坏前缀链 |
| `parallel` | ≥ `--chain-gap` | 确保间隔请求能填满并发窗口 |

建议先使用小规模请求验证：

```python
number=100
parallel=32
rate=1
```

确认成功率和请求格式正常后，再逐步提高 `number`、`parallel` 和 `rate`。

12K 数据中长输出请求较多，最大输出达到 `14050 tokens`。设置请求速率时，需要同时考虑长请求带来的连接占用时间，避免使用过大的 `parallel` 和 `rate` 导致网关主动断开连接。

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
| `--cache-hit-rate` | `0.25` | 每条请求中计划共享的前缀比例，范围 `[0, 1)` |
| `--gsm8k-path` | 空 | 本地 GSM8K JSONL；不指定则尝试从 Hugging Face 加载 |
| `--tokenizer` | 空 | Hugging Face tokenizer 名称或本地权重目录 |
| `--kv-block-size` | `0` | 大于 0 时将共享前缀和链阶段对齐到完整 KV block |
| `--output` | 自动生成 | 输出 JSONL 文件路径 |
| `--format` | `aisbench` | 输出格式：`aisbench` 或 `evalscope` |
| `--body-salt` | 自动生成 | 批次唯一值，控制语料重排、共享块标记、正文块标记和语料偏移 |
| `--seed` | `42` | 控制请求打乱顺序，并参与共享语料偏移计算 |
| `--chain-buckets` | `42100,84200` | 做渐进链的桶输入长度，逗号分隔；在 `CHAIN_BUCKETS` 常量中已定义阶段的桶直接复用，其余按 `--chain-num-stages` 自动均分 |
| `--chain-num-stages` | `7` | 自动生成阶段数，仅对未在 `CHAIN_BUCKETS` 常量定义阶段的桶生效 |
| `--chain-gap` | `0` (auto) | 链 stage 间间隔的普通请求数；0=自动取 concurrency |
| `--chain-intermediate-output` | `16` | 链中间阶段输出 tokens（priming） |
| `--min-chains` | `3` | 每个链桶的最少链数，不足时自动提升 |
| `--no-chain` | `False` | 禁用渐进式前缀链，降级为 12k 行为 |
| `--rate` | `4.0` | EvalScope rate 参数（生成 run_perf.py 用；渐进链模式必须 >0） |

> 想精细控制某个桶的链阶段长度，改 `gen_data_4k_12k.py` 顶部 `CHAIN_BUCKETS` 常量；想快速增删哪些桶做链，用 `--chain-buckets`。

## 渐进链工作原理详解

### 1. 名额分配与链数提升

`--num-requests 3660` 时，各链桶按占比分配链数，`--min-chains 3` 保证每个链桶至少 3 条链：

```
42100 桶 (5.23%) = 191 链 (无需提升)
84200 桶 (0.34%) = 12 链  (无需提升)
普通请求 = 3660 - 191 - 12 = 3457
链请求 = (191 + 12) × 7 阶段 = 1421 (含 priming = 203 × 6 = 1218)
总请求 = 3457 + 1421 = 4878
```

### 2. 每桶 token 序列构造

```
42100 桶: full_seq = [shared_prefix (10496 tok)] [\n] [chain_segments (~31603 tok)] = 42100 tok
84200 桶: full_seq = [shared_prefix (20992 tok)] [\n] [chain_segments (~63207 tok)] = 84200 tok
                ↑ 固定长度 = 25% × 桶长度, block 对齐
```

### 3. 阶段截断 (block 对齐)

```
42100 桶: Stage i = decode(full_seq[: floor(stage_i / 128) × 128])
   [5888, 11904, 17920, 23936, 29952, 35968, 41984]
84200 桶:
   [11904, 23936, 35968, 48000, 59904, 71936, 84096]
```

### 4. 前缀验证

```python
for i in range(6):
    stage_i_tokens = tok.encode(stage_texts[i])
    stage_next_tokens = tok.encode(stage_texts[i+1])
    assert stage_next_tokens[:len(stage_i_tokens)] == stage_i_tokens
```

### 5. 交错排列 (所有链桶统一)

```
[gap×normal], 链0_s0, 链1_s0, ..., 链202_s0,    # 203条链的Stage 0 (42100桶191条 + 84200桶12条)
[gap×normal], 链0_s1, 链1_s1, ..., 链202_s1,    # 203条链的Stage 1
...
[gap×normal], 链0_s6, 链1_s6, ..., 链202_s6,    # 203条链的Stage 6 (目标)
[remaining normal]
```

`gap = concurrency`（默认），确保 Stage i 的 prefill 完成（写缓存）后 Stage i+1 才到达。

### 6. 链定位公式

生成数据集（`--num-requests 3660 --concurrency 128`）后，交错排列的结构固定，可用公式定位任意链的任意阶段在 jsonl 文件中的行号：

```
链 k 的 stage i 行号(1-based) = 2 + (2 + num_chains) × i + k + 1
                              = 2 + 205 × i + k + 1
                              = 3 + 205 × i + k
```

其中 `num_chains = 203`（42100 桶 191 + 84200 桶 12），`gap = 2`（auto 计算的最小值）。

- **42100 桶链**：k = 0..190
- **84200 桶链**：k = 191..202

示例（84200 桶第一条链，k=191）：

| Stage | 公式 | 行号 | max_tokens |
|:---:|---|:---:|---:|
| s0 | 3 + 205×0 + 191 | 194 | 16 |
| s1 | 3 + 205×1 + 191 | 399 | 16 |
| s2 | 3 + 205×2 + 191 | 604 | 16 |
| s3 | 3 + 205×3 + 191 | 809 | 16 |
| s4 | 3 + 205×4 + 191 | 1014 | 16 |
| s5 | 3 + 205×5 + 191 | 1219 | 16 |
| s6 | 3 + 205×6 + 191 | 1424 | 14050 |

同一链相邻 stage 间隔 205 行（= gap 2 + 链总数 203）。

### 7. 实际生成的 max_tokens 分布

`--num-requests 3660` 生成后，jsonl 文件的 `max_tokens` 分布：

```
max_tokens   数量    来源
16           1218    priming (203链 × 6阶段)
10           21      普通请求 (55输入桶)
55           12      普通请求 (330输入桶)
110          37      普通请求 (660输入桶)
220          329     普通请求 (1320输入桶)
440          474     普通请求 (2630输入桶)
870          574     普通请求 (5266输入桶)
1760         1038    普通请求 (10500输入桶)
3500         972     普通请求 (21060输入桶)
7000         191     42100桶链 target
14050        12      84200桶链 target
─────────────────────
总计         4878    (3457普通 + 1421链阶段)
```

## 降级模式

使用 `--no-chain` 可禁用渐进式前缀链，降级为与 ds-v3.2-12k 完全一致的行为：

```bash
python gen_data_4k_12k.py --no-chain --format evalscope
```

此时 84200 桶使用普通请求（含 `[body-<batch>-<rid>-<block>]` 标记），无 priming，无交错排列，`run_perf.py` 可使用 `rate=-1`。

## 与 ds-v3.2-12k 和 ds-v4-3k 的区别

| 维度 | ds-v3.2-12k | ds-v4-3k | **ds-v4-12k** |
|---|---|---|---|
| 目标模型 | DeepSeek V3.2 | DeepSeek V4 Flash | DeepSeek V4 Flash |
| 平均输入 | ~12349 tok | ~3038 tok | ~12349 tok |
| 默认 cache 率 | 25% | 15% | 25% |
| 批次隔离 | 块级标记 | 单一 req 标记 | 块级标记（同 12k） |
| 42100 桶处理 | 单一请求 | — | — | **7 阶渐进链** |
| 84200 桶处理 | 单一请求 | — | 7 阶渐进链 | **7 阶渐进链** |
| 70000 桶处理 | — | 7 阶渐进链 | — | — |
| 缓存命中机制 | 仅 25% 共享前缀 | 15% 共享 + 链内前缀命中 | 25% 共享 + 链内前缀命中 | **25% 共享 + 双桶链内前缀命中** |
| priming 请求 | 无 | 每链 6 个 | 每链 6 个 |
| run_perf rate | 可用 -1 | 必须 > 0 | 必须 > 0 |
| run_perf max_tokens | None (建议) | 必须 None | 必须 None |
| run_perf number | = num-requests | = num-requests + priming | = num-requests + priming |

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

检查渐进链阶段（应看到 `max_tokens: 16` 的 priming 请求，以及两档 target）：

```bash
jq -r '.max_tokens' <generated.jsonl> | sort -n | uniq -c
# 3660 请求数时应包含:
#  1218  16        ← (191+12)链 × 6 priming
#   191  7000      ← 42100 桶链 × 1 target
#    12  14050     ← 84200 桶链 × 1 target
```

定位一条完整链的 7 个阶段行号（用链定位公式）：

```bash
# 84200 桶第一条链 (k=191), 7 个阶段的行号:
# s0=194, s1=399, s2=604, s3=809, s4=1014, s5=1219, s6=1424
# 同一链相邻 stage 间隔 205 行 (= gap 2 + 链总数 203)
for line in 194 399 604 809 1014 1219 1424; do
  sed -n "${line}p" <generated.jsonl> | jq -r '.max_tokens'
done
# 应输出: 16 16 16 16 16 16 14050
```

验证链的前缀关系（Stage i 是 Stage i+1 的前缀）：

```bash
python3 -c "
import json
from transformers import AutoTokenizer
lines = [json.loads(l) for l in open('<generated.jsonl>') if l.strip()]
tok = AutoTokenizer.from_pretrained('./deepseek-v4-flash-tokenizer', trust_remote_code=True)
stages = [lines[p-1]['messages'][0]['content'] for p in [194,399,604,809,1014,1219,1424]]
tok_stages = [tok.encode(s, add_special_tokens=False) for s in stages]
for i in range(6):
    ok = tok_stages[i+1][:len(tok_stages[i])] == tok_stages[i]
    print(f's{i}→s{i+1}: {\"✓\" if ok else \"✗ 漂移\"} ({len(tok_stages[i])}/{len(tok_stages[i])} token)')
"
```

对比两批数据文件是否不同：

```bash
shasum -a 256 batch-A.jsonl batch-B.jsonl
```

## 常见问题

### 渐进链 priming 请求生成了 14050/7000 tokens

检查 `run_perf.py` 中的 `max_tokens`。必须为 `None`，不能是固定值。`None` 让数据集中每条请求自带的 `max_tokens` 生效（priming=16，42100 桶 target=7000，84200 桶 target=14050）。

### 渐进链缓存命中率低

可能原因：

- `rate=-1`（闭环并发）：stage 时序不可控，后阶在前阶 prefill 完成前到达。改为 `rate=N`（N>0）。
- `--chain-gap` 过小：stage 间隔不足，前阶 prefill 未完成。保持 `--chain-gap=0`（auto=concurrency）或增大。
- `--kv-block-size` 与服务端不一致：阶段边界不对齐，缓存部分命中。确认服务端 KV block 大小。
- 服务端未开启 Prefix Caching。
- 服务端缓存不足，早期 stage 被驱逐。
- 不同链请求被路由到不共享缓存的不同实例。

### tokenizer 被当成在线模型仓库

使用完整绝对路径：

```bash
--tokenizer "/apps/models/DeepSeek-V4-Flash"
```

### 出现 `Connection reset by peer`

通常表示网关、代理或模型服务主动断开连接。先降低压力：

```python
number=100
parallel=32
rate=1
```

## 注意事项

- 长度为脚本 tokenizer 测得的用户文本长度，服务端 Chat Template 可能增加额外输入 token。
- 最长档为 `84200 input + 14050 output`，不含 Chat Template 时总长度已经达到 `98250 tokens`。
- 服务端最大上下文长度需要大于最长输入、输出和 Chat Template token 的总和。
- `ignore_eos=true` 需要服务端兼容，否则可能被忽略或返回参数错误。
- 渐进链的 priming 请求（16 tokens 输出）会略微改变整体输出 token 分布，但影响 <1%。
- 不要把 API Key 明文提交到脚本或代码仓库，推荐通过环境变量传入。
- 渐进链模式是**缓存命中基准测试**，不是对话保真度测试。真实 agent 对话有助手响应、工具调用、角色标记等额外 token，本方案未模拟这些。
