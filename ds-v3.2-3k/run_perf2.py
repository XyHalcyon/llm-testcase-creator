# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 逐请求输出长度)。
改下面 model/url/tokenizer_path 后: python run_perf.py
关键: max_tokens=None —— EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值,
      设 None 才让数据里每行自带的 max_tokens 生效(evalscope 1.9.0 实测通过)。
"""
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="deepseek",
    api="openai",
    url="http://localhost:8098/v1/chat/completions",
    dataset="mixed",
    dataset_path='/apps/xhy/dataset/ds-v3.2-3k/gsm8k_3k_c287_cache15_20260731_014648_8436fbc8.jsonl',
    tokenizer_path="/apps/models/DeepSeek-V3.2-w8a8-mtp-QuaRot/",
    number=2870,
    parallel=287,
    rate=2.1,           # 闭环并发; 想按 QPS 到达改成 rate=<req/s> 并调大 parallel
    max_tokens=120000,   # 别改: None 才逐请求生效
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
