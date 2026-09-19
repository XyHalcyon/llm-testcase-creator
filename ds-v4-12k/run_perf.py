# -*- coding: utf-8 -*-
"""正式压测 runner (先跑 run_perf_warmup.py 预热链前缀, 再跑本文件)。
改下面 model/url/tokenizer_path 后: python run_perf.py

关键参数说明:
  - max_tokens=None: 必须为 None, 让数据里每行自带的 max_tokens 生效。
    (EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值。evalscope 1.9.0 实测通过)
  - 预热桶长请求的 TTFT 依赖预热文件已写入的前缀缓存 (前提: 同一服务实例, 缓存未驱逐)。
"""
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="<your-served-model-name>",
    api="openai",
    url="http://<HOST>:<PORT>/v1/chat/completions",
    dataset="mixed",
    dataset_path='/workspace/llm-testcase-creator/ds-v4-12k/gsm8k_4k_12k_c128_cache25_20260919_164926_fcf9e5e1.jsonl',
    tokenizer_path="<tokenizer 或权重目录>",
    number=2161,
    parallel=128,
    rate=4.000000,
    max_tokens=None,      # 别改: None 才逐请求生效 (预热桶 target=636/837/528)
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
