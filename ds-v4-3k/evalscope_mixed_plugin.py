# -*- coding: utf-8 -*-
"""gen_data_4k_unique.py 配套的 EvalScope 自定义数据集 plugin(注册 dataset 名 'mixed')。
每行是一个完整请求 dict(messages + 逐请求 max_tokens + ignore_eos)，兼容旧的
prompt 格式并自动转换为 Chat Completions 所需的 messages。
不要直接用 CLI 跑: EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值 ——
用同目录的 run_perf.py(内置 max_tokens=None)启动。
若 import 路径因版本不同报错, 按你的 EvalScope 调整下面两行 import。
"""
import json
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('mixed')
class MixedDataset(DatasetPluginBase):
    def build_messages(self):
        for line in self.dataset_line_by_line(self.query_parameters.dataset_path):
            line = line.strip()
            if line:
                request = json.loads(line)
                if "messages" not in request and "prompt" in request:
                    request["messages"] = [
                        {"role": "user", "content": request.pop("prompt")}
                    ]
                yield request
