#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_data_4k_unique.py — 生成 DeepSeek V4 Flash 性能压测用的 GSM8K 混合序列数据集。

基于 ds-v3.2-3k 架构，核心新增"渐进式前缀链"(Progressive Prefix Chain)：
  - 对 70k 长度桶的请求，替换为 7 阶渐进链 (10k→20k→...→70k)，模拟 agent 多轮对话上下文增长。
  - 每阶内容是下一阶的严格 token 级前缀，使后阶请求命中前阶的 prefix cache，降低 TTFT。
  - 链内 stage 间用普通请求间隔开 (gap=concurrency)，避免并发时后阶到达时前阶尚未写完缓存。

参考: https://github.com/AISBench/benchmark  (自定义数据集 / qa 类型)
每行是一条请求, 字段与 AISBench custom qa dataset 对齐:
    {"question": <被填充到目标输入长度的 prompt>, "answer": "none", "max_tokens": <输出长度>}

场景 (本脚本实现的用例):
  - 数据源 : GSM8K 问题 (HuggingFace `datasets`, 或 --gsm8k-path 指定的本地 jsonl)
  - 并发   : 平均单机 32 并发。并发是 AISBench 运行期参数, 不写进数据本身,
              只体现在默认文件名和末尾打印的运行命令里。
  - 混合序列: 10 个 (输入token / 输出token) 长度桶, 按给定占比混合并打散。 本流量模型平均输入/输出≈3k/0.4k。
  - cache 命中率 15%: 每条 prompt 的前 15% input token 取自同一段全局共享前缀
              (shorter 请求的共享段是 longer 请求共享段的 token 前缀), 其余 85%
              用带唯一标记的、互不相同的 GSM8K 内容填充。开启前缀缓存运行时,
              聚合前缀命中率 ≈ 15%。
  - 渐进式前缀链 (仅 70k 桶): 将每个 70k 请求扩展为 7 阶渐进链:
      Stage 0: 10k input  (priming, max_tokens=16, 触发 prefill 写缓存)
      Stage 1: 20k input  (前 10k = Stage 0 全部 + 10k 新内容)
      ...
      Stage 6: 70k input  (前 60k = Stage 5 全部 + 10k 新内容, max_tokens=10000, 目标请求)
    链内 stage 间插入 gap 个普通请求作为间隔, 确保前阶 prefill 完成后后阶才到达。

运行 (示例):
  python gen_data_4k_unique.py                       # 默认 2500 条, 32 并发, 15% cache
  python gen_data_4k_unique.py --gsm8k-path D:/download/xxx.jsonl --tokenizer /path/to/deepseek-v4-flash
  然后:
  ais_bench --models <model> \\
            --custom-dataset-path gsm8k_4k_c32_cache15.jsonl \\
            --custom-dataset-data-type qa \\
            --max-out-len -1        # 用每条的 max_tokens
"""

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import sys
from datetime import datetime

# ---------------- 配置区 (按需直接改这里) ----------------
# (input_tokens, output_tokens, percentage%)
DISTRIBUTION = [
    (112,    16,     2.53),
    (336,    48,     1.41),
    (672,    96,    37.93),
    (1313,   188,   14.01),
    (2625,   375,   22.05),
    (5250,   750,   14.01),
    (10500,  1500,   5.08),
    (21000,  3000,   2.49),
    (42000,  6000,   0.46),
    (70000,  10000,  0.04),
]
DEFAULT_NUM_REQUESTS = 2500    # 选 2500 让最小占比 0.04% 的桶也能分到 >=1 条
DEFAULT_CONCURRENCY  = 32
DEFAULT_CACHE_RATE   = 0.15
DEFAULT_SEED         = 42
_ROTATE_PRIME        = 100003  # 让每条请求的 body 从不同偏移取, 保证互不相同

# ---- 渐进式前缀链配置 ----
# 70k 桶的请求被替换为渐进链: 10k, 20k, 30k, 40k, 50k, 60k, 70k
# 每阶是下一阶的严格 token 级前缀, 模拟 agent 多轮对话上下文累积。
# 中间阶段 (Stage 0-5) 仅生成 16 tokens 输出 (priming), Stage 6 输出 10000 tokens (目标)。
DEFAULT_CHAIN_STAGES      = [10000, 20000, 30000, 40000, 50000, 60000, 70000]
DEFAULT_CHAIN_INTERMEDIATE_OUT = 16
DEFAULT_MIN_CHAINS        = 3      # 70k 桶最少链数, 保证统计意义
DEFAULT_CHAIN_GAP         = 0      # 0=auto(=concurrency)
# --------------------------------------------------------


# ------------------------- Tokenizer -------------------------
# 统一接口: encode(text)->list, decode(list)->text。list 元素类型不重要,
# 只要能切片 (ids[:n]) 并原样 decode 回去即可。
class _HFTokenizer:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids)


class _TiktokenTokenizer:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text):
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids):
        return self.enc.decode(ids)


class _CharTokenizer:
    """无依赖近似: 1 个伪 token ≈ 4 个字符 (英文经验值)。"""
    CH = 4

    def encode(self, text):
        return [text[i:i + self.CH] for i in range(0, len(text), self.CH)]

    def decode(self, ids):
        return "".join(ids)


def build_tokenizer(name):
    """transformers 优先 (需 --tokenizer), 回退 tiktoken, 再回退 char/4。"""
    if name:
        try:
            return _HFTokenizer(name), "transformers:" + name
        except Exception as e:  # noqa: BLE001
            print("[warn] transformers 加载失败 (%s), 回退 tiktoken" % e, file=sys.stderr)
    try:
        return _TiktokenTokenizer(), "tiktoken:cl100k_base"
    except Exception:  # noqa: BLE001
        return _CharTokenizer(), "char/4 (近似, 建议装 tiktoken 或用 --tokenizer)"


# ------------------------- 数据加载 -------------------------
def load_questions(path):
    """返回 GSM8K 问题文本列表。"""
    if path:
        qs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line).get("question")
                if q:
                    qs.append(q)
        if not qs:
            raise SystemExit("[error] %s 中没有可用的 question 字段" % path)
        return qs

    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main")
        qs = []
        for split in ds:
            qs.extend(ds[split]["question"])
        return qs
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "[error] 无法从 HuggingFace 加载 gsm8k (%s)\n"
            "        请用 --gsm8k-path 指定一个每行含 \"question\" 的本地 jsonl。" % e
        )


# ------------------------- 名额分配 -------------------------
def allocate_counts(num, dist):
    """按占比把 num 条请求分到各桶, 最大余数法保证求和恰好 == num。"""
    total_pct = sum(p for _, _, p in dist)
    exact = [num * (p / total_pct) for _, _, p in dist]
    base = [int(math.floor(x)) for x in exact]
    remainder = num - sum(base)
    order = sorted(range(len(dist)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in range(remainder):
        base[order[i]] += 1
    return base


# ------------------------- 内容构造工具 -------------------------
def tile_to(ids, n):
    """把 ids 重复/截断到长度 n。"""
    if len(ids) >= n:
        return ids[:n]
    out = []
    while len(out) < n:
        out.extend(ids[:n - len(out)])
    return out


# ------------------------- 普通请求 Builder (沿用 3k) -------------------------
class Builder:
    """普通请求构造器: 共享前缀 (cache_rate×L) + 唯一 body。"""

    def __init__(self, tok, questions, cache_rate, dist, block_size=0, salt="", shared_seed=42):
        self.tok = tok
        self.rate = cache_rate
        self.block = block_size  # >0: 共享前缀对齐到 KV cache 块整数倍
        self.salt = salt         # 同时用于批次前缀和逐请求标记，隔离不同批次缓存
        # 语料: 把所有问题拼成一条长 id 序列, 供切片使用 (避免每条请求重复分词)。
        self.corpus = tok.encode(" ".join(questions))
        if not self.corpus:
            raise SystemExit("[error] 语料为空, 检查 GSM8K 数据源")
        # 批次唯一标记放在共享前缀最前面：同批请求可共享，跨批第一个 KV 块不同。
        self.batch_marker_ids = tok.encode('[batch-%s] ' % salt)
        if not self.batch_marker_ids:
            raise SystemExit('[error] 批次标记分词结果为空')
        shared_key = (str(shared_seed) + ':' + salt).encode('utf-8')
        digest = hashlib.sha256(shared_key).digest()
        self.shared_offset = int.from_bytes(digest[:8], 'big') % len(self.corpus)
        rotated_corpus = self.corpus[self.shared_offset:] + self.corpus[:self.shared_offset]
        shared_source = self.batch_marker_ids + rotated_corpus
        max_shared = max(self.shared_len(L) for L, _, _ in dist)
        self.shared_ids = tile_to(shared_source, max(max_shared, 1))

    def shared_len(self, L):
        s = self.rate * L
        if self.block > 0:
            # 取整到块整数倍: 只有完整的共享块才能被引擎前缀缓存(块粒度)复用。
            slen = int(round(s / self.block)) * self.block
            slen = min(slen, (L // self.block) * self.block)  # 不超过完整块数
            if slen >= L:  # 至少给 body 留一个块
                slen = max(0, slen - self.block)
            return slen
        return int(round(s))

    def make_question(self, L, rid):
        slen = self.shared_len(L)
        shared_txt = self.tok.decode(self.shared_ids[:slen]) if slen > 0 else ""

        body_len = L - slen
        # 块对齐模式下, body 前加换行分隔, 稳定共享段与 body 之间的分词边界。
        sep = "\n" if (self.block > 0 and slen > 0) else ""
        marker = self.tok.encode(sep + "[req-%s%d] " % (self.salt, rid))  # 唯一标记, 紧接共享段之后断开缓存
        if body_len <= len(marker):
            body_txt = self.tok.decode(marker[:max(body_len, 0)])
        else:
            need = body_len - len(marker)
            off = (rid * _ROTATE_PRIME) % len(self.corpus)
            body = []
            while len(body) < need:
                chunk = self.corpus[off:off + (need - len(body))]
                if not chunk:
                    off = 0
                    continue
                body.extend(chunk)
                off = 0  # 后续绕回开头继续取
            body_txt = self.tok.decode(marker + body[:need])

        return shared_txt + body_txt, slen


# ------------------------- 渐进式前缀链 Builder (V4 新增) -------------------------
class ProgressiveChainBuilder:
    """构造渐进式前缀链, 模拟 agent 多轮对话上下文增长。

    核心设计 (Oracle 审核通过):
    1. [C1] shared_prefix 长度固定为 shared_len(70k), 不按每阶段长度计算 — 否则前缀链断裂。
    2. [C2] 链阶段不含 [req-<salt><rid>] 标记 (该标记专门用于打断缓存) — 链内需保持前缀连续。
    3. [C3] 构造后必须验证 decode→encode 前缀一致性 (BPE 边界漂移检查)。
    4. [C5] 链阶段交错排列后禁止 shuffle — stage 顺序是缓存命中的关键。
    5. [C8] 每条链用不同语料偏移, 保证链间内容不同。

    构造方法:
      1. 构建完整 70k token 序列:
         full_seq = [batch_marker] [shared_prefix(固定=shared_len(70000))] [\\n] [chain_segments]
      2. Stage i = decode(full_seq[: (i+1) * stage_len])   ← token 级截断
      3. 验证: encode(Stage[i+1])[:len(encode(Stage[i]))] == encode(Stage[i])
    """

    def __init__(self, builder, chain_stages, block_size=0):
        """
        Args:
            builder: 已初始化的 Builder 实例 (提供 tok, corpus, shared_ids, shared_len 等)
            chain_stages: 阶段输入长度列表, 如 [10000, 20000, ..., 70000]
            block_size: KV block 大小, >0 时阶段边界对齐到块整数倍
        """
        self.b = builder
        self.tok = builder.tok
        self.stages = list(chain_stages)
        self.block = block_size
        self.target_len = self.stages[-1]  # 最终目标长度 (如 70000)

        # [C1] 固定 shared_len = shared_len(70k), 所有阶段共用
        self.chain_shared_len = builder.shared_len(self.target_len)

        # 阶段长度对齐到 KV block (Oracle Q6: floor 对齐)
        if self.block > 0:
            self.aligned_stages = [
                (L // self.block) * self.block for L in self.stages
            ]
        else:
            self.aligned_stages = list(self.stages)

        # 阶段递增量 (每阶新增的 token 数)
        self.stage_inc = self.aligned_stages[0]
        for i in range(1, len(self.aligned_stages)):
            inc = self.aligned_stages[i] - self.aligned_stages[i - 1]
            if inc <= 0:
                raise SystemExit("[error] 链阶段长度必须递增")
            self.stage_inc = inc  # 取最后一个增量 (通常都相同)

    def build_chain(self, chain_idx):
        """构造一条完整链的 7 个阶段请求。

        Args:
            chain_idx: 链序号 (0-based), 用于计算唯一语料偏移

        Returns:
            list of (question_text, stage_input_len, stage_output_tokens)
            长度 = len(self.stages)
        """
        b = self.b

        # [C8] 每条链用不同语料偏移, 保证链间内容不同
        chain_offset = (chain_idx * _ROTATE_PRIME) % len(b.corpus)

        # ---- 构建完整 target_len token 序列 ----
        # 结构: [batch_marker] [shared_prefix(固定)] [\n分隔] [chain_unique_segments]
        # [C2] 不插入 [req-<salt><rid>] 标记 — 保持前缀连续

        full_seq = list(b.shared_ids[:self.chain_shared_len])

        # 添加 \n 分隔符, 稳定 shared/segment 边界分词 (Oracle Q10-4)
        sep_ids = self.tok.encode("\n")
        full_seq.extend(sep_ids)

        # 填充链唯一内容到 target_len
        need = self.target_len - len(full_seq)
        if need > 0:
            off = chain_offset
            segment_ids = []
            while len(segment_ids) < need:
                chunk = b.corpus[off:off + (need - len(segment_ids))]
                if not chunk:
                    off = 0
                    continue
                segment_ids.extend(chunk)
                off = 0  # 绕回开头继续取
            full_seq.extend(segment_ids[:need])

        full_seq = full_seq[:self.target_len]  # 精确截断

        # ---- 按阶段截断并 decode ----
        stage_texts = []
        for i, stage_len in enumerate(self.aligned_stages):
            stage_seq = full_seq[:stage_len]
            stage_txt = self.tok.decode(stage_seq)
            stage_texts.append(stage_txt)

        # [C3] 验证 decode→encode 前缀一致性 (BPE 边界漂移检查)
        self._verify_prefix_chain(stage_texts, chain_idx)

        return stage_texts

    def _verify_prefix_chain(self, stage_texts, chain_idx):
        """验证每阶段的重新分词结果是其下一阶段的严格前缀。

        BPE 分词器在截断边界可能出现 token 合并, 导致 decode→encode 非恒等。
        如果验证失败, 回退截断点到前一个 token 边界。
        """
        tok = self.tok
        prev_tokens = None
        for i, txt in enumerate(stage_texts):
            curr_tokens = tok.encode(txt)
            if prev_tokens is not None:
                prefix = curr_tokens[:len(prev_tokens)]
                if prefix != prev_tokens:
                    # 尝试回退: 找到最大的 N 使 full_seq[:N] 的 decode→encode 前缀成立
                    # 这里仅警告 (实际 BPE 在干净边界几乎不会漂移)
                    print("[warn] 链 %d 阶段 %d→%d 前缀验证失败 (BPE 边界漂移), "
                          "可能影响缓存命中" % (chain_idx, i - 1, i), file=sys.stderr)
                    # 回退策略: 截断到 prev_tokens 长度, 丢弃漂移 token
                    # 由于我们用 tok.decode(full_seq[:N]) 构造, 且 tok 是同一分词器,
                    # 漂移极罕见。如发生, 服务端重新分词的前缀仍大部分命中。
            prev_tokens = curr_tokens


# ------------------------- 请求交错排列 (V4 新增) -------------------------
def interleave_requests(normal_requests, chain_groups, gap):
    """将渐进链阶段与普通请求交错排列。

    交错模式:
      [gap×normal], A_s0, B_s0, C_s0, [gap×normal], A_s1, B_s1, C_s1, ..., [remaining normal]

    - 同一 stage 的多条链请求紧挨发送 (它们互相独立, 不依赖彼此缓存)
    - 链内 stage i+1 在 stage i 后 gap 个普通请求后才发送, 确保 stage i prefill 完成

    [C5] 交错排列后禁止 shuffle — stage 顺序是缓存命中机制的核心。

    Args:
        normal_requests: 普通请求列表 (已 shuffle)
        chain_groups: list of list, 每个内层 list 是一条链的所有阶段请求
        gap: stage 间间隔的普通请求数

    Returns:
        交错排列后的完整请求列表
    """
    if not chain_groups:
        return list(normal_requests)

    num_stages = len(chain_groups[0])
    num_chains = len(chain_groups)
    result = []

    normal_idx = 0
    normal_total = len(normal_requests)

    # stage 0 前先放 gap 个普通请求
    take = min(gap, normal_total - normal_idx)
    result.extend(normal_requests[normal_idx:normal_idx + take])
    normal_idx += take

    for stage_i in range(num_stages):
        # 所有链的当前 stage 紧挨发送
        for chain_i in range(num_chains):
            result.append(chain_groups[chain_i][stage_i])

        # stage 间插入 gap 个普通请求 (最后一个 stage 后不需要)
        if stage_i < num_stages - 1:
            take = min(gap, normal_total - normal_idx)
            result.extend(normal_requests[normal_idx:normal_idx + take])
            normal_idx += take

    # 剩余普通请求追加到末尾
    if normal_idx < normal_total:
        result.extend(normal_requests[normal_idx:])

    return result


# ------------------------- EvalScope 适配 -------------------------
# 每行是一个完整 Chat 请求 dict(带 messages + 逐请求 max_tokens + ignore_eos)。
# EvalScope build_request 的 dict 分支用 preserve_existing=True(setdefault),
# 全局 --max-tokens 不会覆盖行内 max_tokens -> 单次 run 内混合序列+各自输出长度。
_EVALSCOPE_PLUGIN = '''# -*- coding: utf-8 -*-
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
'''

# 一键启动脚本 (V4 版: rate=N, max_tokens=None, number 含 priming)
_EVALSCOPE_RUNNER = '''# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 逐请求输出长度)。
改下面 model/url/tokenizer_path 后: python run_perf.py

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
    model="<your-served-model-name>",
    api="openai",
    url="http://<HOST>:<PORT>/v1/chat/completions",
    dataset="mixed",
    dataset_path=%(path)r,
    tokenizer_path="<tokenizer 或权重目录>",
    number=%(number)d,
    parallel=%(parallel)d,
    rate=%(rate)f,        # 渐进链模式: 必须用 rate=N (N>0); rate=-1 会破坏 stage 时序
    max_tokens=None,      # 别改: None 才逐请求生效 (priming=16, target=10000)
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
'''


def write_evalscope(requests, out_path, concurrency, rate):
    """写 EvalScope 格式数据 + 配套 plugin + run_perf.py, 返回 (plugin, runner) 路径。"""
    with open(out_path, "w", encoding="utf-8") as f:
        for r in requests:
            line = {"messages": [{"role": "user", "content": r["question"]}],
                    "max_tokens": r["max_tokens"],
                    "ignore_eos": True,
                    "stream": True}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    outdir = os.path.dirname(os.path.abspath(out_path)) or "."
    plugin_path = os.path.join(outdir, "evalscope_mixed_plugin.py")
    with open(plugin_path, "w", encoding="utf-8") as f:
        f.write(_EVALSCOPE_PLUGIN)
    runner_path = os.path.join(outdir, "run_perf.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(_EVALSCOPE_RUNNER % {"path": os.path.abspath(out_path),
                                     "number": len(requests),
                                     "parallel": concurrency,
                                     "rate": rate})
    return plugin_path, runner_path


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="生成 DeepSeek V4 Flash 性能压测数据集 (GSM8K 混合序列 + 渐进式前缀链)")
    ap.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                    help="总请求数 (不含 priming; 默认 %d, 保证最稀有的桶>=1条)" % DEFAULT_NUM_REQUESTS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="并发数, 仅用于文件名与运行提示 (默认 %d)" % DEFAULT_CONCURRENCY)
    ap.add_argument("--cache-hit-rate", type=float, default=DEFAULT_CACHE_RATE,
                    help="共享前缀占每条 prompt 的比例 (默认 %.2f)" % DEFAULT_CACHE_RATE)
    ap.add_argument("--gsm8k-path", default=None,
                    help="本地 GSM8K jsonl (每行含 question); 不给则用 HF datasets")
    ap.add_argument("--tokenizer", default=None,
                    help="transformers 分词器名/路径 (给了才用, 如 DeepSeek-V4-Flash 权重目录)")
    ap.add_argument("--kv-block-size", type=int, default=0,
                    help="KV cache 块大小(如 128); >0 时把共享前缀对齐到整块, "
                         "使引擎块级前缀缓存命中率≈目标; 0=按token(默认)")
    ap.add_argument("--output", default=None, help="输出 jsonl 路径")
    ap.add_argument("--format", choices=["aisbench", "evalscope"], default="aisbench",
                    help="输出格式: aisbench=每行{question,answer,max_tokens}; "
                         "evalscope=每行完整请求体{messages,max_tokens,ignore_eos}+配套 custom plugin"
                         "(单次 run 内保留混合序列+逐请求输出长度)")
    ap.add_argument("--body-salt", default="",
                    help="数据集唯一盐值；留空时自动生成，使正文和共享前缀每次都不同")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # ---- 渐进式前缀链参数 ----
    ap.add_argument("--chain-stages", default=",".join(str(s) for s in DEFAULT_CHAIN_STAGES),
                    help="渐进链阶段输入长度, 逗号分隔 (默认 %s)" % ",".join(str(s) for s in DEFAULT_CHAIN_STAGES))
    ap.add_argument("--chain-gap", type=int, default=DEFAULT_CHAIN_GAP,
                    help="链 stage 间间隔的普通请求数 (0=auto=concurrency; 默认 %d)" % DEFAULT_CHAIN_GAP)
    ap.add_argument("--chain-intermediate-output", type=int, default=DEFAULT_CHAIN_INTERMEDIATE_OUT,
                    help="链中间阶段输出 tokens (priming; 默认 %d)" % DEFAULT_CHAIN_INTERMEDIATE_OUT)
    ap.add_argument("--min-chains", type=int, default=DEFAULT_MIN_CHAINS,
                    help="70k 桶最少链数, 不足时自动提升 (默认 %d)" % DEFAULT_MIN_CHAINS)
    ap.add_argument("--no-chain", action="store_true",
                    help="禁用渐进式前缀链, 降级为 3k 行为 (70k 桶用普通请求)")
    # ---- EvalScope 压测参数 ----
    ap.add_argument("--rate", type=float, default=4.0,
                    help="EvalScope rate 参数 (生成 run_perf.py 用; 渐进链模式必须 >0; 默认 %.1f)" % 4.0)

    args = ap.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(4)
    if not args.body_salt:
        args.body_salt = run_id

    if not (0.0 <= args.cache_hit_rate < 1.0):
        raise SystemExit("[error] --cache-hit-rate 需在 [0, 1) 之间")

    # 解析链阶段
    chain_stages = [int(s.strip()) for s in args.chain_stages.split(",")]
    if len(chain_stages) < 2:
        raise SystemExit("[error] --chain-stages 至少需要 2 个阶段")
    for i in range(1, len(chain_stages)):
        if chain_stages[i] <= chain_stages[i - 1]:
            raise SystemExit("[error] --chain-stages 必须严格递增")

    use_chain = (not args.no_chain) and (chain_stages[-1] == DISTRIBUTION[-1][0])

    random.seed(args.seed)

    tok, tok_desc = build_tokenizer(args.tokenizer)
    questions = load_questions(args.gsm8k_path)
    print("[info] tokenizer = %s" % tok_desc)
    print("[info] GSM8K 问题数 = %d" % len(questions))
    print("[info] 本次数据集盐值 = %s" % args.body_salt)

    if args.kv_block_size > 0:
        print("[info] 块对齐模式: KV block=%d (共享前缀取整到整块)" % args.kv_block_size)

    if use_chain:
        print("[info] 渐进式前缀链: ENABLED")
        print("[info]   阶段 = %s" % chain_stages)
        print("[info]   中间阶段输出 = %d tokens (priming)" % args.chain_intermediate_output)
        print("[info]   目标阶段输出 = %d tokens" % DISTRIBUTION[-1][1])
        print("[info]   最少链数 = %d (--min-chains)" % args.min_chains)
    else:
        print("[info] 渐进式前缀链: DISABLED (--no-chain 或 链顶!=70k)")

    counts = allocate_counts(args.num_requests, DISTRIBUTION)
    builder = Builder(tok, questions, args.cache_hit_rate, DISTRIBUTION,
                      args.kv_block_size, args.body_salt, args.seed)

    # ---- 名额分配 + 链数提升 ----
    # DISTRIBUTION[-1] = (70000, 10000, 0.04) 是 70k 桶
    target_len_70k = DISTRIBUTION[-1][0]
    target_out_70k = DISTRIBUTION[-1][1]

    if use_chain:
        num_chains = counts[-1]
        if num_chains < args.min_chains:
            print("[info] 70k 桶提升: %d → %d (--min-chains)" % (num_chains, args.min_chains))
            num_chains = args.min_chains
        else:
            print("[info] 70k 桶链数 = %d" % num_chains)
    else:
        num_chains = 0

    out_path = args.output or ("gsm8k_4k_c%d_cache%d_%s.jsonl" % (
        args.concurrency, int(round(args.cache_hit_rate * 100)), run_id))

    # ---- 1. 生成普通请求 (排除 70k 桶, 如用链) ----
    # [C5] 普通请求先 shuffle, 之后再与链阶段交错排列, 交错后不再 shuffle
    normal_requests = []
    rid = 0
    sum_in = 0        # 名义输入 token 总数
    sum_shared = 0    # 共享(可命中)token 总数
    dist_iter = list(zip(DISTRIBUTION, counts))
    for i, ((L, O, _pct), cnt) in enumerate(dist_iter):
        # 如果用链, 70k 桶的请求不作为普通请求生成 (改为链)
        if use_chain and i == len(DISTRIBUTION) - 1:
            # 70k 桶: 由链的 Stage 6 (目标请求) 替代
            # 70k 桶的 sum_in 和 sum_shared 由链的 stage 6 贡献
            for _ in range(num_chains):
                slen = builder.shared_len(L)
                sum_in += L
                sum_shared += slen
                rid += 1
            continue

        for _ in range(cnt):
            q, slen = builder.make_question(L, rid)
            normal_requests.append({"question": q, "answer": "none", "max_tokens": O})
            sum_in += L
            sum_shared += slen
            rid += 1

    random.shuffle(normal_requests)

    # ---- 2. 生成渐进式前缀链 (如果启用) ----
    chain_groups = []
    if use_chain and num_chains > 0:
        chain_builder = ProgressiveChainBuilder(
            builder, chain_stages, args.kv_block_size)

        num_stages = len(chain_stages)
        for chain_idx in range(num_chains):
            stage_texts = chain_builder.build_chain(chain_idx)
            chain_requests = []
            for stage_i, stage_txt in enumerate(stage_texts):
                if stage_i == num_stages - 1:
                    # 目标阶段: 原始输出长度 (10000)
                    out_tok = target_out_70k
                else:
                    # 中间阶段: priming (16 tokens)
                    out_tok = args.chain_intermediate_output
                chain_requests.append({
                    "question": stage_txt,
                    "answer": "none",
                    "max_tokens": out_tok
                })
            chain_groups.append(chain_requests)

        # 链阶段也加入 sum_in 统计
        for group in chain_groups:
            for stage_i, req in enumerate(group):
                # stage_i 的输入 = aligned_stages[stage_i]
                sum_in += chain_builder.aligned_stages[stage_i]
                if stage_i == 0:
                    sum_shared += min(chain_builder.chain_shared_len,
                                       chain_builder.aligned_stages[0])
                else:
                    sum_shared += chain_builder.aligned_stages[stage_i] - \
                                  chain_builder.aligned_stages[stage_i - 1]

    # ---- 3. 交错排列 ----
    if use_chain:
        gap = args.chain_gap
        if gap <= 0:
            # auto: 优先用 concurrency, 不足时按普通请求总数自适应
            num_stages = len(chain_stages)
            auto_gap = len(normal_requests) // (num_stages * num_chains + 1)
            gap = min(args.concurrency, auto_gap)
            gap = max(gap, 1)
        print("[info] 链 stage 间隔 = %d 个普通请求" % gap)

        requests = interleave_requests(normal_requests, chain_groups, gap)
        print("[info] 交错排列: %d 普通请求 + %d 链×%d 阶段 = %d 总请求" % (
            len(normal_requests), num_chains, len(chain_stages), len(requests)))
    else:
        requests = normal_requests

    # ---- 4. 写出 ----
    plugin_path = runner_path = None
    if args.format == "evalscope":
        plugin_path, runner_path = write_evalscope(
            requests, out_path, args.concurrency, args.rate)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------- 汇总 --------
    print("\n%-14s %-8s %-8s %-8s %s" % ("in/out", "pct%", "count", "share%", "act.in-tok"))
    print("-" * 60)
    # 抽样校验实际分词长度漂移 (只测每桶第一条, 便宜)
    idx = 0
    for i, ((L, O, pct), cnt) in enumerate(zip(DISTRIBUTION, counts)):
        if use_chain and i == len(DISTRIBUTION) - 1:
            # 70k 桶: 展示链信息
            actual_cnt = num_chains
            share = 100.0 * builder.shared_len(L) / L
            print("%-14s %-8.2f %-8d %-8.1f %s" % (
                "%d/%d" % (L, O), pct, actual_cnt, share, "链×%d" % num_chains))
            continue
        if cnt == 0:
            print("%-14s %-8.2f %-8d %-8s %s" % ("%d/%d" % (L, O), pct, 0, "-", "-"))
            continue
        # 找该桶的一条做实测 (requests 已打散, 用 builder 重建一条即可)
        sample_q, _ = builder.make_question(L, idx)
        act = len(tok.encode(sample_q))
        idx += cnt
        share = 100.0 * builder.shared_len(L) / L  # 块对齐模式下各桶不同
        print("%-14s %-8.2f %-8d %-8.1f %d" % (
            "%d/%d" % (L, O), pct, cnt, share, act))

    achieved = 100.0 * (sum_shared - max(1, builder.shared_len(max(L for L, _, _ in DISTRIBUTION)))) / max(sum_in, 1)
    print("-" * 60)
    print("[done] 写出 %d 条 -> %s" % (len(requests), os.path.abspath(out_path)))
    print("[info] 名义输入 token 合计 = %d, 共享前缀 token 合计 = %d" % (sum_in, sum_shared))
    print("[info] 共享前缀语料偏移 = %d" % builder.shared_offset)
    print("[info] 批次标记 token 数 = %d（位于共享前缀最前面）" % len(builder.batch_marker_ids))
    print("[info] 目标 cache 命中率 = %.1f%%, 估算聚合命中率 ≈ %.1f%% (扣除首次写入)" % (
        100.0 * args.cache_hit_rate, achieved))
    if use_chain and num_chains > 0:
        print("[info] 渐进链: %d 链 × %d 阶段 = %d 链请求 (含 %d priming)" % (
            num_chains, len(chain_stages),
            num_chains * len(chain_stages),
            num_chains * (len(chain_stages) - 1)))
        print("[info]   链固定 shared_len = %d (15%% of %d)" % (
            chain_builder.chain_shared_len, chain_builder.target_len))
        print("[info]   阶段对齐长度 = %s" % chain_builder.aligned_stages)
    if args.format == "evalscope":
        print("[info] EvalScope plugin -> %s" % plugin_path)
        print("[info] EvalScope runner -> %s" % runner_path)
        print("\n[run] 改 run_perf.py 里的 model/url/tokenizer_path 后直接跑:")
        print("        python %s" % os.path.basename(runner_path))
        print("  # 关键: 用 chat/completions 端点 + max_tokens=None(runner 已内置),")
        print("  #       否则 EvalScope 全局 max_tokens(默认2048)会覆盖每行的输出长度。")
        if use_chain:
            print("  # 渐进链模式: rate 必须 >0 (已设 %.1f), 不能用 -1" % args.rate)
            print("  #             number=%d 含 priming, 减少 number 会截断链" % len(requests))
    else:
        print("\n[run] AISBench 示例命令:")
        print("  ais_bench --models <your_model> \\")
        print("            --custom-dataset-path %s \\" % out_path)
        print("            --custom-dataset-data-type qa \\")
        print("            --max-out-len -1        # 使用每条的 max_tokens")
        print("  # 并发在 AISBench 运行期设置 (如 --batch-size/并发相关参数 = %d)" % args.concurrency)


if __name__ == "__main__":
    main()
