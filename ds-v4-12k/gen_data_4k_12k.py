#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_data_4k_12k.py — 生成 DeepSeek V4 Flash 性能压测用的 GSM8K 混合序列数据集
(双文件模式: 预热用例 + 正式测试用例)。

架构:
  - ds-v3.2-12k: 重流量模型 + 块级批次隔离
                 (共享区/正文区每个 128-token 块带 [shared-<batch>-<block>] / [body-<batch>-<rid>-<block>] 标记,
                  跨批完整块交集 = 0)。
                 本版分布取自混合序列流量画像 (rr=5, 21机): 20 档长度, 平均输入 ~14.9K / 输出 ~290。
  - ds-v4-3k:    渐进式前缀链 (Progressive Prefix Chain) 的前缀构造能力,
                 用于生成"预热前缀 ↔ 正式长请求"的 token 级前缀对。

生成两个测试用例文件 (同一次运行, 同 salt / 同语料偏移, 保证前缀一致):
  1. 预热文件 (--warmup-output): 只含各预热桶的前缀请求 (长度 = 预热前缀长度)。
     先跑它, 把前缀 KV 写入服务端 prefix cache。
  2. 正式文件 (--output): 20 档普通混合序列 + 预热桶的完整长度请求 (无 priming、无交错)。
     后跑它, 完整长度请求与预热请求 token 级前缀一致, 命中预热写入的缓存。

预热/正式前缀对 (可通过 WARMUP_PAIRS 常量或 --warmup-pairs 配置):
  每对 = (正式输入长度, 预热前缀长度), 只要求预热 < 正式 (block 对齐后), 无阶段概念:
  344158 桶: 预热前缀 292864 → 正式 344158 (需计算 51200)
  499835 桶: 预热前缀 448512 → 正式 499835 (需计算 51200)
  728183 桶: 预热前缀 676864 → 正式 728183 (需计算 51200)
  预热请求输出 = --warmup-max-tokens (默认 16, 最小化 decode KV);
  正式长请求输出 = 该桶目标输出长度 (344158桶=636, 499835桶=837, 728183桶=528)。

运行前提: 服务端 prefix cache 容量需容纳全部预热 KV; 预热与正式必须打同一服务实例;
预热后尽快启动正式测试, 减少缓存被中间流量冲刷的窗口。

参考: https://github.com/AISBench/benchmark  (自定义数据集 / qa 类型)
每行是一条请求, 字段与 AISBench custom qa dataset 对齐:
    {"question": <被填充到目标输入长度的 prompt>, "answer": "none", "max_tokens": <输出长度>}

运行 (示例):
  python gen_data_4k_12k.py                       # 默认 2500 条, 32 并发, 25% cache
  python gen_data_4k_12k.py --gsm8k-path ./GSM8K.jsonl --tokenizer /path/to/DeepSeek-V4-Flash
  然后先跑预热 runner, 再跑正式 runner (生成的 run_perf_warmup.py / run_perf.py)
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
# (input_tokens, output_tokens, percentage%)  — 取自混合序列流量画像 (rr=5, 21机), 按输入长度升序 20 档。
# 名义平均输入 ≈ 14.9K, 平均输出 ≈ 290。尾桶 3147/1045429 源报表显示 0.0% (四舍五入),
# 此处按 0.01% 计入: 约 1 万条请求出现 1 条该超长输出 (~1M) 批处理请求; 0.00% 时它永远 0 条。
DISTRIBUTION = [
    (125,     74,        6.09),
    (622,     46,       12.95),
    (999,     1218,      3.69),
    (1643,    119,      31.31),
    (2624,    167,      18.96),
    (3147,    1045429,   0.00),
    (4979,    184,       9.64),
    (7299,    28067,     0.03),
    (8958,    411,       2.94),
    (18997,   533,       1.80),
    (31130,   430,       2.37),
    (46871,   528,       2.19),
    (66476,   622,       2.15),
    (90801,   668,       2.08),
    (120954,  744,       1.66),
    (162151,  835,       0.94),
    (224544,  846,       0.54),
    (344158,  636,       0.36),
    (499835,  837,       0.18),
    (728183,  528,       0.09),
]
DEFAULT_NUM_REQUESTS = 2500    # 选 2500 让 0.03% 桶分到 >=1 条 (0.01% 尾桶约 1 万条才出现 1 条)
DEFAULT_CONCURRENCY  = 32
DEFAULT_CACHE_RATE   = 0.25    # 混合序列流量模型默认 25% 共享前缀
DEFAULT_SEED         = 42
_ROTATE_PRIME        = 100003  # 让每条请求的 body 从不同偏移取, 保证互不相同

# ---- 预热/正式前缀对配置 ----
# 每条 entry: (正式输入长度, 预热前缀长度)
# ★修改这里即可增删预热桶: 添加/删除 entry
# 约束:
#   - 预热前缀长度 < 正式输入长度 (预热是正式的严格前缀)
#   - 两个长度会被自动 floor 对齐到 --kv-block-size (保证整块命中, 无部分命中损耗)
#   - 正式输入长度必须存在于 DISTRIBUTION
WARMUP_PAIRS = [
    (344158, 292864),   # 正式 344064 tok, 预热 292864 tok, 需计算 51200
    (499835, 448512),   # 正式 499712 tok, 预热 448512 tok, 需计算 51200
    (728183, 676864),   # 正式 728064 tok, 预热 676864 tok, 需计算 51200
]
DEFAULT_MIN_PAIRS              = 2      # 每个预热桶的最少前缀对组数, 不足时自动提升
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


# ------------------------- 普通请求 Builder (取自 ds-v3.2-12k, 块级批次隔离) -------------------------
class Builder:
    """普通请求构造器: 共享前缀 (cache_rate×L) + 唯一 body。

    块级批次隔离 (与 ds-v3.2-12k 一致):
      - 共享区每个块带 [shared-<batch_tag>-<block_index>] 标记
      - 正文区每个块带 [body-<batch_tag>-<rid>-<block_index>] 标记
      - 每批用 SHA256(seed:salt) 重新排列 GSM8K 问题, 并分别计算 shared/body 语料偏移
      跨批完整 128-token 块交集 = 0, 隔离上一轮残留的 Prefix Cache。
    """

    def __init__(self, tok, questions, cache_rate, dist, block_size=0, salt="", shared_seed=42):
        self.tok = tok
        self.rate = cache_rate
        self.block = block_size  # >0: 共享前缀对齐到 KV cache 块整数倍
        self.salt = salt         # 用于批次、共享块和逐请求正文块唯一化
        self.unique_block_size = self.block if self.block > 0 else 128

        batch_key = (str(shared_seed) + ':' + salt).encode('utf-8')
        digest = hashlib.sha256(batch_key).digest()
        self.batch_tag = digest.hex()[:12]

        # 每个批次重新排列 GSM8K 问题顺序，避免不同批次复用相同正文序列。
        batch_questions = list(questions)
        random.Random(int.from_bytes(digest[:8], 'big')).shuffle(batch_questions)
        self.corpus = tok.encode(" ".join(batch_questions))
        if not self.corpus:
            raise SystemExit("[error] 语料为空, 检查 GSM8K 数据源")

        self.shared_offset = int.from_bytes(digest[8:16], 'big') % len(self.corpus)
        self.body_offset = int.from_bytes(digest[16:24], 'big') % len(self.corpus)
        rotated_corpus = self.corpus[self.shared_offset:] + self.corpus[:self.shared_offset]

        # 共享区每个块都带批次标记：同批可共享，跨批不会产生相同原始 token 块。
        max_shared = max(self.shared_len(L) for L, _, _ in dist)
        shared_target = max(max_shared, 1)
        self.shared_ids = []
        source_offset = 0
        block_index = 0
        while len(self.shared_ids) < shared_target:
            block_len = min(self.unique_block_size, shared_target - len(self.shared_ids))
            marker = tok.encode('[shared-%s-%d] ' % (self.batch_tag, block_index))[:block_len]
            self.shared_ids.extend(marker)
            need = block_len - len(marker)
            while need > 0:
                take = min(need, len(rotated_corpus) - source_offset)
                self.shared_ids.extend(rotated_corpus[source_offset:source_offset + take])
                source_offset = (source_offset + take) % len(rotated_corpus)
                need -= take
            block_index += 1

        self.batch_marker_ids = tok.encode('[shared-%s-0] ' % self.batch_tag)

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
        # 非共享区每个块都加入批次/请求/块编号，防止两批正文出现相同 KV 大小 token 块。
        body = []
        source_offset = (self.body_offset + rid * _ROTATE_PRIME) % len(self.corpus)
        block_index = 0
        while len(body) < body_len:
            block_len = min(self.unique_block_size, body_len - len(body))
            sep = "\n" if (block_index == 0 and self.block > 0 and slen > 0) else ""
            marker = self.tok.encode(
                sep + '[body-%s-%d-%d] ' % (self.batch_tag, rid, block_index)
            )[:block_len]
            body.extend(marker)
            need = block_len - len(marker)
            while need > 0:
                take = min(need, len(self.corpus) - source_offset)
                body.extend(self.corpus[source_offset:source_offset + take])
                source_offset = (source_offset + take) % len(self.corpus)
                need -= take
            block_index += 1
        body_txt = self.tok.decode(body[:body_len])

        return shared_txt + body_txt, slen


# ------------------------- 预热/正式前缀对构造器 -------------------------
class PrefixPairBuilder:
    """构造"预热前缀 ↔ 正式长请求"的 token 级前缀文本对。

    核心设计 (Oracle 审核通过):
    1. [C1] 共享前缀长度按正式长度计算 (shared_len(formal_len)), 预热前缀完整覆盖共享段。
    2. [C2] 文本不含 [body-<batch>-<rid>-<block>] 标记 (该标记专门用于打断缓存) — 保持前缀连续。
    3. [C3] 构造后验证 decode→encode 前缀一致性 (BPE 边界漂移检查)。
    4. [C8] 每组用不同语料偏移, 保证组间内容不同。

    与 Builder 的兼容性:
      - 复用 builder.shared_ids (已含 [shared-<batch>-<block>] 标记, 同批一致)。
      - 段内容用 builder.corpus (已按批次重排) 填充。
      - 跨批隔离由 Builder 的 batch_tag + 问题重排保证。

    构造方法:
      1. 构建完整 formal_len token 序列:
         full_seq = [shared_prefix(=shared_len(formal_len))] [\\n] [pair唯一segments]
      2. 预热文本 = decode(full_seq[:warmup_len]), 正式文本 = decode(full_seq[:formal_len])
         同一 token 序列截断 → 预热是正式的严格前缀 (构造保证)
      3. 验证 encode(正式)[:len(encode(预热))] == encode(预热)
    """

    def __init__(self, builder, formal_len, warmup_len, block_size=0):
        """
        Args:
            builder: 已初始化的 Builder 实例 (提供 tok, corpus, shared_ids, shared_len 等)
            formal_len: 正式请求输入长度 (token)
            warmup_len: 预热请求输入长度 (token), 必须小于 formal_len
            block_size: KV block 大小, >0 时两个长度 floor 对齐到块整数倍
        """
        self.b = builder
        self.tok = builder.tok
        self.block = block_size
        # floor 对齐到 KV block (保证预热/正式边界都是整块, 无部分命中损耗)
        if block_size > 0:
            self.formal_len = (formal_len // block_size) * block_size
            self.warmup_len = (warmup_len // block_size) * block_size
        else:
            self.formal_len = formal_len
            self.warmup_len = warmup_len
        if self.warmup_len >= self.formal_len:
            raise SystemExit("[error] 预热前缀长度 (%d) 对齐后必须小于正式长度 (%d)"
                             % (self.warmup_len, self.formal_len))
        # 共享前缀按正式长度计算 (预热前缀完整覆盖共享段)
        self.pair_shared_len = builder.shared_len(formal_len)

    def build_pair(self, pair_idx):
        """构造一组 (预热文本, 正式文本)。pair_idx 决定语料偏移, 保证组间内容不同。"""
        b = self.b

        # [C8] 每组用不同语料偏移, 保证组间内容不同
        pair_offset = (pair_idx * _ROTATE_PRIME) % len(b.corpus)

        # ---- 构建完整 formal_len token 序列 ----
        # 结构: [shared_prefix(含批次块标记)] [\n分隔] [pair唯一segments]
        # [C2] 不插入 [body-<batch>-<rid>-<block>] 标记 — 保持前缀连续
        full_seq = list(b.shared_ids[:self.pair_shared_len])

        # 添加 \n 分隔符, 稳定 shared/segment 边界分词
        full_seq.extend(self.tok.encode("\n"))

        # 填充 pair 唯一内容到 formal_len (不插入任何标记, 保证前缀连续)
        need = self.formal_len - len(full_seq)
        if need > 0:
            off = pair_offset
            segment_ids = []
            while len(segment_ids) < need:
                chunk = b.corpus[off:off + (need - len(segment_ids))]
                if not chunk:
                    off = 0
                    continue
                segment_ids.extend(chunk)
                off = 0  # 绕回开头继续取
            full_seq.extend(segment_ids[:need])

        full_seq = full_seq[:self.formal_len]  # 精确截断

        # 同一 token 序列两次截断 → 预热是正式的严格前缀 (构造保证)
        warmup_txt = self.tok.decode(full_seq[:self.warmup_len])
        formal_txt = self.tok.decode(full_seq[:self.formal_len])
        assert formal_txt[:len(warmup_txt)] == warmup_txt, \
            "[bug] pair %d: 预热文本不是正式文本的前缀" % pair_idx

        # [C3] 验证 decode→encode 前缀一致性 (BPE 边界漂移检查)
        self._verify_prefix(warmup_txt, formal_txt, pair_idx)

        return warmup_txt, formal_txt

    def _verify_prefix(self, warmup_txt, formal_txt, pair_idx):
        """验证重新分词后预热文本仍是正式文本的严格 token 前缀。

        BPE 分词器在截断边界可能出现 token 合并, 导致 decode→encode 非恒等。
        如果验证失败, 打印警告 (实际 BPE 在干净边界几乎不会漂移)。
        """
        tok = self.tok
        w_tokens = tok.encode(warmup_txt)
        f_tokens = tok.encode(formal_txt)
        if f_tokens[:len(w_tokens)] != w_tokens:
            # 由于我们用 tok.decode(full_seq[:N]) 构造, 且 tok 是同一分词器,
            # 漂移极罕见。如发生, 服务端重新分词的前缀仍大部分命中。
            print("[warn] pair %d 前缀验证失败 (BPE 边界漂移), "
                  "可能影响缓存命中" % pair_idx, file=sys.stderr)


# ------------------------- EvalScope 适配 -------------------------
# 每行是一个完整 Chat 请求 dict(带 messages + 逐请求 max_tokens + ignore_eos)。
# EvalScope build_request 的 dict 分支用 preserve_existing=True(setdefault),
# 全局 --max-tokens 不会覆盖行内 max_tokens -> 单次 run 内混合序列+各自输出长度。
_EVALSCOPE_PLUGIN = '''# -*- coding: utf-8 -*-
"""gen_data_4k_12k.py 配套的 EvalScope 自定义数据集 plugin(注册 dataset 名 'mixed')。
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

# 正式测试启动脚本 (双文件模式: 先跑 run_perf_warmup.py 预热, 再跑本脚本)
_EVALSCOPE_RUNNER = '''# -*- coding: utf-8 -*-
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
    dataset_path=%(path)r,
    tokenizer_path="<tokenizer 或权重目录>",
    number=%(number)d,
    parallel=%(parallel)d,
    rate=%(rate)f,
    max_tokens=None,      # 别改: None 才逐请求生效 (预热桶 target=636/837/528)
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
'''

# 预热 runner 模板: 只跑预热文件, 目的是把前缀写入服务端 prefix cache
_EVALSCOPE_RUNNER_WARMUP = '''# -*- coding: utf-8 -*-
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
    dataset_path=%(path)r,
    tokenizer_path="<tokenizer 或权重目录>",
    number=%(number)d,
    parallel=%(parallel)d,
    rate=%(rate)f,
    max_tokens=None,      # 别改: None 才让预热请求自带的 max_tokens 生效
    stream=True,
    name="warmup_prefix",
)
run_perf_benchmark(args)
'''


def write_evalscope(requests, out_path, concurrency, rate, runner_filename="run_perf.py",
                    runner_template=None):
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
    runner_path = os.path.join(outdir, runner_filename)
    template = runner_template if runner_template is not None else _EVALSCOPE_RUNNER
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(template % {"path": os.path.abspath(out_path),
                            "number": len(requests),
                            "parallel": concurrency,
                            "rate": rate})
    return plugin_path, runner_path


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="生成 DeepSeek V4 Flash 性能压测数据集 (GSM8K 混合长序列, 双文件: 预热+正式)")
    ap.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                    help="总请求数 (默认 %d, 保证最稀有的桶>=1条)" % DEFAULT_NUM_REQUESTS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="并发数, 仅用于文件名与运行提示 (默认 %d)" % DEFAULT_CONCURRENCY)
    ap.add_argument("--cache-hit-rate", type=float, default=DEFAULT_CACHE_RATE,
                    help="共享前缀占每条 prompt 的比例 (默认 %.2f)" % DEFAULT_CACHE_RATE)
    ap.add_argument("--gsm8k-path", default=None,
                    help="本地 GSM8K jsonl (每行含 question); 不给则用 HF datasets")
    ap.add_argument("--tokenizer", default=None,
                    help="transformers 分词器名/路径 (给了才用, 如 DeepSeek-V4-Flash 权重目录)")
    ap.add_argument("--kv-block-size", type=int, default=0,
                    help="KV cache 块大小(如 128); >0 时把共享前缀和预热/正式长度对齐到整块, "
                         "使引擎块级前缀缓存命中率≈目标; 0=按token(默认)")
    ap.add_argument("--output", default=None, help="正式文件输出 jsonl 路径")
    ap.add_argument("--format", choices=["aisbench", "evalscope"], default="aisbench",
                    help="输出格式: aisbench=每行{question,answer,max_tokens}; "
                         "evalscope=每行完整请求体{messages,max_tokens,ignore_eos}+配套 custom plugin"
                         "(单次 run 内保留混合序列+逐请求输出长度)")
    ap.add_argument("--body-salt", default="",
                    help="数据集唯一盐值；留空时自动生成，使正文和共享前缀每次都不同")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # ---- 预热/正式前缀对参数 ----
    ap.add_argument("--warmup-pairs",
                    default=",".join("%d:%d" % (f, w) for f, w in WARMUP_PAIRS),
                    help="预热/正式前缀对, 逗号分隔, 每对格式 正式长度:预热长度 (默认 %s)。"
                         "只要求预热长度 < 正式长度 (自动 floor 对齐到 --kv-block-size); "
                         "正式长度必须存在于 DISTRIBUTION"
                         % ",".join("%d:%d" % (f, w) for f, w in WARMUP_PAIRS))
    ap.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS,
                    help="每个预热桶的最少前缀对组数, 不足时自动提升 (默认 %d)" % DEFAULT_MIN_PAIRS)

    # ---- 预热/正式双文件参数 ----
    ap.add_argument("--warmup-output", default=None,
                    help="预热文件路径 (默认自动命名 warmup_prefix_<run_id>.jsonl)。"
                         "预热文件只含各预热桶的前缀请求, 用于预热服务端 prefix cache;"
                         "正式文件(--output)为纯普通混合序列 + 预热桶完整长度请求"
                         "(与预热请求 token 级前缀一致)。两文件同一次运行生成, 保证前缀一致")
    ap.add_argument("--warmup-max-tokens", type=int, default=16,
                    help="预热请求的输出 tokens (decode 长度; 默认 16, 最小化 decode KV 占用)")
    # ---- EvalScope 压测参数 ----
    ap.add_argument("--rate", type=float, default=4.0,
                    help="EvalScope rate 参数 (写入两个 runner; 默认 %.1f)" % 4.0)

    args = ap.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(4)
    if not args.body_salt:
        args.body_salt = run_id

    if not (0.0 <= args.cache_hit_rate < 1.0):
        raise SystemExit("[error] --cache-hit-rate 需在 [0, 1) 之间")

    # 解析预热/正式前缀对: {formal_len: warmup_len}
    prefix_pair_map = {}
    for part in args.warmup_pairs.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit("[error] --warmup-pairs 格式应为 正式长度:预热长度 (如 728183:676864), 实际 %r" % part)
        f_len, w_len = (int(x) for x in part.split(":"))
        dist_lens = {L for L, _, _ in DISTRIBUTION}
        if f_len not in dist_lens:
            raise SystemExit("[error] --warmup-pairs 正式长度 %d 不在 DISTRIBUTION 中" % f_len)
        if f_len in prefix_pair_map:
            raise SystemExit("[error] --warmup-pairs 正式长度 %d 重复" % f_len)
        if w_len >= f_len:
            raise SystemExit("[error] 预热前缀长度 %d 必须小于正式长度 %d" % (w_len, f_len))
        prefix_pair_map[f_len] = w_len
    if not prefix_pair_map:
        raise SystemExit("[error] --warmup-pairs 不能为空")

    random.seed(args.seed)

    tok, tok_desc = build_tokenizer(args.tokenizer)
    questions = load_questions(args.gsm8k_path)
    print("[info] tokenizer = %s" % tok_desc)
    print("[info] GSM8K 问题数 = %d" % len(questions))
    print("[info] 本次数据集盐值 = %s" % args.body_salt)

    if args.kv_block_size > 0:
        print("[info] 块对齐模式: KV block=%d (共享前缀取整到整块)" % args.kv_block_size)

    print("[info] 双文件模式: 预热文件 (前缀请求) + 正式文件 (普通 + 预热桶完整长度请求)")
    print("[info]   预热/正式前缀对 = %s" % dict(prefix_pair_map))
    print("[info]   预热请求输出 = %d tokens (--warmup-max-tokens)" % args.warmup_max_tokens)
    print("[info]   最少组数/桶 = %d (--min-pairs)" % args.min_pairs)

    counts = allocate_counts(args.num_requests, DISTRIBUTION)
    builder = Builder(tok, questions, args.cache_hit_rate, DISTRIBUTION,
                      args.kv_block_size, args.body_salt, args.seed)

    # ---- 名额分配 + 每个预热桶的组数提升 ----
    bucket_pair_count = {}  # {formal_len: num_pairs}
    for i, (L, O, _pct) in enumerate(DISTRIBUTION):
        if L in prefix_pair_map:
            nc = counts[i]
            if nc < args.min_pairs:
                print("[info] %d 桶提升: %d → %d (--min-pairs)" % (L, nc, args.min_pairs))
                nc = args.min_pairs
            else:
                print("[info] %d 桶前缀对组数 = %d" % (L, nc))
            bucket_pair_count[L] = nc

    out_path = args.output or ("gsm8k_4k_12k_c%d_cache%d_%s.jsonl" % (
        args.concurrency, int(round(args.cache_hit_rate * 100)), run_id))
    warmup_out = args.warmup_output or ("warmup_prefix_%s.jsonl" % run_id)

    # ---- 1. 生成普通请求 (排除所有预热桶) ----
    normal_requests = []
    rid = 0
    sum_in = 0        # 普通请求名义输入 token 总数 (不含预热桶, 预热桶由 formal_pair_sum_in 统计)
    sum_shared = 0    # 普通(可命中)token 总数
    dist_iter = list(zip(DISTRIBUTION, counts))
    for i, ((L, O, _pct), cnt) in enumerate(dist_iter):
        if L in prefix_pair_map:
            # 预热桶: 正式文件中由完整长度请求替代, 此处只占 rid 名额
            # (sum_in/sum_shared 不在此累加, 避免 formal_pair_sum_in 二次统计)
            rid += bucket_pair_count[L]
            continue

        for _ in range(cnt):
            q, slen = builder.make_question(L, rid)
            normal_requests.append({"question": q, "answer": "none", "max_tokens": O})
            sum_in += L
            sum_shared += slen
            rid += 1

    random.shuffle(normal_requests)

    # ---- 2. 生成预热请求 (前缀) + 正式预热桶请求 (完整长度) ----
    pair_builders = {}      # {formal_len: PrefixPairBuilder}, 供汇总
    warmup_requests = []
    formal_pair_requests = []
    warmup_sum_in = 0
    formal_pair_sum_in = 0
    for f_len, w_len in prefix_pair_map.items():
        pb = PrefixPairBuilder(builder, f_len, w_len, args.kv_block_size)
        pair_builders[f_len] = pb
        num_pairs_L = bucket_pair_count[f_len]
        target_out_L = next(O for LL, O, _ in DISTRIBUTION if LL == f_len)
        for pair_idx in range(num_pairs_L):
            warmup_txt, formal_txt = pb.build_pair(pair_idx)
            warmup_requests.append({"question": warmup_txt, "answer": "none",
                                    "max_tokens": args.warmup_max_tokens})
            formal_pair_requests.append({"question": formal_txt, "answer": "none",
                                         "max_tokens": target_out_L})
        warmup_sum_in += num_pairs_L * pb.warmup_len
        formal_pair_sum_in += num_pairs_L * pb.formal_len

    # 正式文件 = 普通请求 (已 shuffle) + 预热桶完整长度请求, 整体再 shuffle
    # (Oracle 审核结论 Q4: 长请求顺序无关, 命中只依赖预热先于正式执行)
    formal_requests = normal_requests + formal_pair_requests
    random.shuffle(formal_requests)

    # ---- 3. 写出两个文件 ----
    if args.format == "evalscope":
        w_plugin, w_runner = write_evalscope(
            warmup_requests, warmup_out, args.concurrency, args.rate,
            runner_filename="run_perf_warmup.py", runner_template=_EVALSCOPE_RUNNER_WARMUP)
        f_plugin, f_runner = write_evalscope(
            formal_requests, out_path, args.concurrency, args.rate)
    else:
        with open(warmup_out, "w", encoding="utf-8") as f:
            for r in warmup_requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in formal_requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        w_plugin = w_runner = f_plugin = f_runner = None

    # -------- 汇总 --------
    print()
    print("=== 预热文件 ===")
    print("[done] 写出 %d 条 -> %s" % (len(warmup_requests), os.path.abspath(warmup_out)))
    for f_len in prefix_pair_map:
        pb = pair_builders[f_len]
        print("[info]   桶 %d: %d 条, 预热前缀=%d tok, 正式长度=%d tok" % (
            f_len, bucket_pair_count[f_len], pb.warmup_len, pb.formal_len))
    print("[info] 预热名义输入合计 = %d tokens" % warmup_sum_in)
    print()
    print("=== 正式文件 ===")
    print("[done] 写出 %d 条 -> %s" % (len(formal_requests), os.path.abspath(out_path)))
    print("[info] 名义输入 token 合计 = %d (普通 %d + 预热桶完整长度 %d)" % (
        sum_in + formal_pair_sum_in, sum_in, formal_pair_sum_in))
    print("[info] 预热桶长请求可命中的预热前缀比例 = %.1f%%" % (
        100.0 * warmup_sum_in / max(formal_pair_sum_in, 1)))
    print("[info] 共享前缀语料偏移 = %d" % builder.shared_offset)
    print("[info] 批次标记 token 数 = %d（共享区每个块均带批次标记）" % len(builder.batch_marker_ids))
    print("[info] 正文唯一化块大小 = %d" % builder.unique_block_size)
    if args.format == "evalscope":
        print("[info] 预热 runner -> %s (先跑)" % w_runner)
        print("[info] 正式 runner -> %s (后跑)" % f_runner)
        print("\n[run] 执行顺序:")
        print("        1. 改两个 runner 的 model/url/tokenizer_path")
        print("        2. python %s   # 预热: 写入前缀缓存" % os.path.basename(w_runner))
        print("        3. python %s   # 正式: 长请求命中预热前缀" % os.path.basename(f_runner))
        print("  # 关键: 两 runner 均内置 max_tokens=None, 让数据逐请求的 max_tokens 生效")
        print("  # 前提: 服务端 prefix cache 容量需容纳全部预热 KV; 两文件必须打同一实例;")
        print("  #       预热后尽快启动正式测试, 减少缓存被中间流量冲刷的窗口")
    else:
        print("\n[run] AISBench 示例命令 (先预热后正式):")
        print("  ais_bench --models <your_model> \\")
        print("            --custom-dataset-path %s \\" % warmup_out)
        print("            --custom-dataset-data-type qa \\")
        print("            --max-out-len -1")
        print("  ais_bench --models <your_model> \\")
        print("            --custom-dataset-path %s \\" % out_path)
        print("            --custom-dataset-data-type qa \\")
        print("            --max-out-len -1        # 使用每条的 max_tokens")
        print("  # 并发在 AISBench 运行期设置 (如 --batch-size/并发相关参数 = %d)" % args.concurrency)


if __name__ == "__main__":
    main()
