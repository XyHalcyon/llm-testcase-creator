# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 渐进式前缀链 + 逐请求输出长度) - 配置 2。
改下面 model/url/tokenizer_path 后: python run_perf2.py

配置 2: 中速率压测 (rate=4, 渐进链时序仍然可控, 吞吐更高)。
关键参数 (渐进式前缀链模式):
  - max_tokens=None: 必须为 None, 让数据里每行自带的 max_tokens 生效
    (priming=16, 42100桶target=7000, 84200桶target=14050)。
    与 ds-v3.2-12k 的 run_perf2 不同: 那里用 120000 覆盖每行输出长度,
    这里设固定值会使 priming 阶段也生成该长度, 完全失去 priming 意义。
  - rate=N (N>0): 必须用固定速率模式。rate=-1 (闭环并发) 会使 stage 时序不可控,
    后阶可能在前阶 prefill 完成前到达, 导致缓存未命中。
  - number: 必须包含 priming 请求数 (4878 = 3457 普通 + 1421 链阶段)。
"""
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="deepseek-v4-flash",
    api="openai",
    url="http://localhost:8098/v1/chat/completions",
    dataset="mixed",
    dataset_path='/workspace/llm-testcase-creator/ds-v4-12k/gsm8k_4k_12k_c128_cache25_20260910_072644_53b9b8de.jsonl',
    tokenizer_path="/apps/models/DeepSeek-V4-Flash/",
    number=4878,
    parallel=128,
    rate=4,           # 渐进链模式: 必须用 rate=N (N>0); 中速率压测
    max_tokens=None,  # 别改: None 才逐请求生效 (priming=16, target=7000/14050)
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
