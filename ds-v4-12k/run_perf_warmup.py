# -*- coding: utf-8 -*-
"""预热压测 runner: 先跑本文件, 把预热前缀的 KV 写入服务端 prefix cache,
之后正式文件 (run_perf.py) 中对应的长请求才能命中该前缀缓存。

部署前提 (Oracle 审核结论):
  1. 服务端 prefix cache 容量必须能容纳全部预热 KV (~前缀对组数 × 预热前缀长度 token),
     否则预热自逐出, 正式测试无法命中。
  2. 预热与正式必须打到同一个服务实例 (实例重启缓存丢失)。
  3. 预热完成后尽快启动正式测试, 减少中间流量冲刷缓存的窗口。

关键参数: max_tokens=None (预热请求自带 max_tokens=16, 不能被全局值覆盖)。
"""
import evalscope_mixed_plugin  # noqa: F401  注册 dataset 'mixed'
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

args = Arguments(
    model="<your-served-model-name>",
    api="openai",
    url="http://<HOST>:<PORT>/v1/chat/completions",
    dataset="mixed",
    dataset_path='/workspace/llm-testcase-creator/ds-v4-12k/warmup_prefix.jsonl',
    tokenizer_path="<tokenizer 或权重目录>",
    number=14,
    parallel=128,
    rate=4.000000,
    max_tokens=None,      # 别改: None 才让预热请求自带的 max_tokens 生效
    stream=True,
    name="warmup_prefix",
)
run_perf_benchmark(args)
