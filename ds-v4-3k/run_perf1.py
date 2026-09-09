# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 渐进式前缀链 + 逐请求输出长度)。
改下面 model/url/tokenizer_path 后: python run_perf1.py

关键参数说明 (渐进式前缀链模式):
  - max_tokens=None: 必须为 None, 让数据里每行自带的 max_tokens 生效。
    (EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值; 设 120000 会使 priming
    阶段也生成 120k tokens, 完全失去 priming 意义。evalscope 1.9.0 实测通过)
  - rate=N (N>0): 必须用固定速率模式。rate=-1 (闭环并发) 会使 stage 时序不可控,
    后阶可能在前阶 prefill 完成前到达, 导致缓存未命中。
  - number: 必须包含 priming 请求数。减少 number 会截断链阶段, 破坏前缀链。
"""
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="deepseek-v4-flash",
    api="openai",
    url="http://localhost:8099/v1/chat/completions",
    dataset="mixed",
    dataset_path='/apps/dataset/ds-v4-3k/gsm8k_4k_c256_cache15_20260909_120000_a1b2c3d4.jsonl',
    tokenizer_path="/apps/models/DeepSeek-V4-Flash/",
    number=2560,
    parallel=256,
    rate=4,             # 渐进链模式: 必须用 rate=N (N>0); rate=-1 会破坏 stage 时序
    max_tokens=None,    # 别改: None 才逐请求生效 (priming=16, target=10000)
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
