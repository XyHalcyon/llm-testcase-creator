#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_data_4k_12k.py — 生成 DeepSeek V4 Flash 性能压测用的 GSM8K 混合序列数据集 (混合长序列 + 渐进式前缀链)。

融合两套架构:
  - ds-v3.2-12k: 重流量模型 + 块级批次隔离
                 (共享区/正文区每个 128-token 块带 [shared-<batch>-<block>] / [body-<batch>-<rid>-<block>] 标记,
                  跨批完整块交集 = 0)。
                 本版分布取自混合序列流量画像 (rr=5, 21机): 20 档长度, 平均输入 ~14.9K / 输出 ~355。
  - ds-v4-3k:    渐进式前缀链 (Progressive Prefix Chain) — 将最长桶的请求替换为多阶渐进链,
                 每阶是下一阶的严格 token 级前缀, 使后阶请求命中前阶 prefix cache, 降低 TTFT。

本脚本对 344158、499835 和 728183 三个长输入桶启用渐进式前缀链 (可通过 CHAIN_BUCKETS 常量或
--chain-buckets 参数配置哪些桶做链):
  344158 桶: 7 阶链 (49k→98k→147k→196k→245k→294k→344158), 每轮约 49k 新内容
  499835 桶: 7 阶链 (71k→143k→214k→286k→357k→428k→499835), 每轮约 71k 新内容
  728183 桶: 7 阶链 (104k→208k→312k→416k→520k→624k→728183), 每轮约 104k 新内容
  - Stage 0-5: max_tokens=16 (priming, 触发 prefill 写缓存)
  - Stage 6:   max_tokens=该桶目标输出长度 (344158桶=636, 499835桶=837, 728183桶=528)
  每阶是下一阶的严格 token 级前缀, 链内 stage 间插入 gap 个普通请求间隔,
  确保前阶 prefill 完成后后阶才到达, 避免并发乱序破坏命中。

参考: https://github.com/AISBench/benchmark  (自定义数据集 / qa 类型)
每行是一条请求, 字段与 AISBench custom qa dataset 对齐:
    {"question": <被填充到目标输入长度的 prompt>, "answer": "none", "max_tokens": <输出长度>}

运行 (示例):
  python gen_data_4k_12k.py                       # 默认 2500 条, 32 并发, 25% cache
  python gen_data_4k_12k.py --gsm8k-path ./GSM8K.jsonl --tokenizer /path/to/DeepSeek-V4-Flash
  然后:
  ais_bench --models <model> \\
            --custom-dataset-path gsm8k_4k_12k_c32_cache25.jsonl \\
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
# (input_tokens, output_tokens, percentage%)  — 取自混合序列流量画像 (rr=5, 21机), 按输入长度升序 20 档。
# 名义平均输入 ≈ 14.9K, 平均输出 ≈ 355。尾桶 3147/1045429 源报表显示 0.0% (四舍五入),
# 此处按 0.01% 计入: 约 1 万条请求出现 1 条该超长输出 (~1M) 批处理请求; 0.00% 时它永远 0 条。
DISTRIBUTION = [
    (125,     74,        6.09),
    (622,     46,       12.95),
    (999,     1218,      3.69),
    (1643,    119,      31.31),
    (2624,    167,      18.96),
    (3147,    1045429,   0.01),
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

# ---- 渐进式前缀链配置 (多桶) ----
# 哪些输入长度桶做渐进式前缀链处理, 及其链阶段。
# ★修改这里即可增删链桶: 添加/删除 entry。每条 entry: (bucket_input_len, [stage_lengths])
# 约束:
#   - stages 必须严格递增, 且最后一阶 == bucket_input_len
#   - 所有链桶的阶段数必须相同 (交错排列约束)
#   - stage_lengths 会被自动 floor 对齐到 --kv-block-size
# 中间阶段 (非最后) 仅生成 16 tokens 输出 (priming), 最后阶段输出该桶的目标输出长度。
DEFAULT_CHAIN_BUCKETS = [
    (344158, [49165, 98330, 147495, 196660, 245825, 294990, 344158]),  # 每轮 ~49k 新内容
    (499835, [71405, 142810, 214215, 285620, 357025, 428430, 499835]),  # 每轮 ~71k 新内容
    (728183, [104026, 208052, 312078, 416104, 520130, 624156, 728183]), # 每轮 ~104k 新内容
]
DEFAULT_CHAIN_NUM_STAGES        = 7      # CLI --chain-buckets 自动生成阶段时的阶段数
DEFAULT_CHAIN_INTERMEDIATE_OUT = 16
DEFAULT_MIN_CHAINS             = 3      # 每个链桶的最少链数, 不足时自动提升
DEFAULT_CHAIN_GAP              = 0      # 0=auto(=concurrency)
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


# ------------------------- 渐进式前缀链 Builder (取自 ds-v4-3k, 适配 12k Builder) -------------------------
class ProgressiveChainBuilder:
    """构造渐进式前缀链, 模拟 agent 多轮对话上下文增长。

    核心设计 (Oracle 审核通过):
    1. [C1] shared_prefix 长度固定为 shared_len(链桶 target_len), 不按每阶段长度计算 — 否则前缀链断裂。
    2. [C2] 链阶段不含 [body-<batch>-<rid>-<block>] 标记 (该标记专门用于打断缓存) — 链内需保持前缀连续。
    3. [C3] 构造后必须验证 decode→encode 前缀一致性 (BPE 边界漂移检查)。
    4. [C5] 链阶段交错排列后禁止 shuffle — stage 顺序是缓存命中的关键。
    5. [C8] 每条链用不同语料偏移, 保证链间内容不同。

    与 ds-v3.2-12k Builder 的兼容性:
      - 直接复用 builder.shared_ids (已含 [shared-<batch>-<block>] 标记, 同批一致, 不破坏前缀连续)。
      - 链段用 builder.corpus (已按批次重新排列) 填充, 不插入 [body-] 标记。
      - 跨批隔离由 12k Builder 的 batch_tag + 问题重排保证, 链段天然继承。

    构造方法:
      1. 构建完整 target_len (如 728183) token 序列:
         full_seq = [batch_marker(shared块)] [shared_prefix(固定=shared_len(target_len))] [\\n] [chain_segments]
      2. Stage i = decode(full_seq[: (i+1) * stage_len])   ← token 级截断
      3. 验证: encode(Stage[i+1])[:len(encode(Stage[i]))] == encode(Stage[i])
    """

    def __init__(self, builder, chain_stages, block_size=0):
        """
        Args:
            builder: 已初始化的 Builder 实例 (提供 tok, corpus, shared_ids, shared_len 等)
            chain_stages: 阶段输入长度列表, 如 [104026, 208052, ..., 728183]
            block_size: KV block 大小, >0 时阶段边界对齐到块整数倍
        """
        self.b = builder
        self.tok = builder.tok
        self.stages = list(chain_stages)
        self.block = block_size
        self.target_len = self.stages[-1]  # 最终目标长度 (如 728183)

        # [C1] 固定 shared_len = shared_len(target_len), 所有阶段共用
        self.chain_shared_len = builder.shared_len(self.target_len)

        # 阶段长度对齐到 KV block (Oracle Q6: floor 对齐)
        if self.block > 0:
            self.aligned_stages = [
                (L // self.block) * self.block for L in self.stages
            ]
        else:
            self.aligned_stages = list(self.stages)

        # 阶段递增量校验
        for i in range(1, len(self.aligned_stages)):
            if self.aligned_stages[i] <= self.aligned_stages[i - 1]:
                raise SystemExit("[error] 链阶段长度必须递增")

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
        # 结构: [shared_prefix(固定, 含批次块标记)] [\n分隔] [chain_unique_segments]
        # [C2] 不插入 [body-<batch>-<rid>-<block>] 标记 — 保持前缀连续

        full_seq = list(b.shared_ids[:self.chain_shared_len])

        # 添加 \n 分隔符, 稳定 shared/segment 边界分词 (Oracle Q10-4)
        sep_ids = self.tok.encode("\n")
        full_seq.extend(sep_ids)

        # 填充链唯一内容到 target_len (不插入任何标记, 保证前缀连续)
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
                off = 0  # 后续绕回开头继续取
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
        如果验证失败, 打印警告 (实际 BPE 在干净边界几乎不会漂移)。
        """
        tok = self.tok
        prev_tokens = None
        for i, txt in enumerate(stage_texts):
            curr_tokens = tok.encode(txt)
            if prev_tokens is not None:
                prefix = curr_tokens[:len(prev_tokens)]
                if prefix != prev_tokens:
                    # 由于我们用 tok.decode(full_seq[:N]) 构造, 且 tok 是同一分词器,
                    # 漂移极罕见。如发生, 服务端重新分词的前缀仍大部分命中。
                    print("[warn] 链 %d 阶段 %d→%d 前缀验证失败 (BPE 边界漂移), "
                          "可能影响缓存命中" % (chain_idx, i - 1, i), file=sys.stderr)
            prev_tokens = curr_tokens


# ------------------------- 请求交错排列 (取自 ds-v4-3k) -------------------------
def interleave_requests(normal_requests, chain_groups, gap):
    """将渐进链阶段与普通请求交错排列。

    交错模式:
      [gap×normal], A_s0, B_s0, C_s0, [gap×normal], A_s1, B_s1, C_s1, ..., A_s6, B_s6, C_s6, [remaining normal]

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

# 一键启动脚本 (V4 混合序列版: rate=N, max_tokens=None, number 含 priming)
_EVALSCOPE_RUNNER = '''# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 逐请求输出长度)。
改下面 model/url/tokenizer_path 后: python run_perf.py

关键参数说明 (渐进式前缀链模式):
  - max_tokens=None: 必须为 None, 让数据里每行自带的 max_tokens 生效。
    (EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值; 设 837 会使 priming
    阶段也生成该长度, 完全失去 priming 意义。evalscope 1.9.0 实测通过)
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
    max_tokens=None,      # 别改: None 才逐请求生效 (priming=16, target=837/528)
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
        description="生成 DeepSeek V4 Flash 性能压测数据集 (GSM8K 混合长序列 + 渐进式前缀链)")
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
                    help="KV cache 块大小(如 128); >0 时把共享前缀和链阶段对齐到整块, "
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
    ap.add_argument("--chain-buckets",
                    default=",".join(str(b) for b, _ in DEFAULT_CHAIN_BUCKETS),
                    help="做渐进链的桶输入长度, 逗号分隔 (默认 %s)。"
                         "在 CHAIN_BUCKETS 常量中已定义阶段的桶直接复用; "
                         "未定义的桶按 --chain-num-stages 自动均分+block对齐生成阶段。"
                         % ",".join(str(b) for b, _ in DEFAULT_CHAIN_BUCKETS))
    ap.add_argument("--chain-num-stages", type=int, default=DEFAULT_CHAIN_NUM_STAGES,
                    help="自动生成阶段数 (默认 %d), 仅对未在 CHAIN_BUCKETS 常量定义阶段的桶生效"
                         % DEFAULT_CHAIN_NUM_STAGES)
    ap.add_argument("--chain-gap", type=int, default=DEFAULT_CHAIN_GAP,
                    help="链 stage 间间隔的普通请求数 (0=auto=concurrency; 默认 %d)" % DEFAULT_CHAIN_GAP)
    ap.add_argument("--chain-intermediate-output", type=int, default=DEFAULT_CHAIN_INTERMEDIATE_OUT,
                    help="链中间阶段输出 tokens (priming; 默认 %d)" % DEFAULT_CHAIN_INTERMEDIATE_OUT)
    ap.add_argument("--min-chains", type=int, default=DEFAULT_MIN_CHAINS,
                    help="每个链桶的最少链数, 不足时自动提升 (默认 %d)" % DEFAULT_MIN_CHAINS)
    ap.add_argument("--no-chain", action="store_true",
                    help="禁用渐进式前缀链, 降级为普通混合序列行为 (链桶改用普通请求)")
    # ---- EvalScope 压测参数 ----
    ap.add_argument("--rate", type=float, default=4.0,
                    help="EvalScope rate 参数 (生成 run_perf.py 用; 渐进链模式必须 >0; 默认 %.1f)" % 4.0)

    args = ap.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(4)
    if not args.body_salt:
        args.body_salt = run_id

    if not (0.0 <= args.cache_hit_rate < 1.0):
        raise SystemExit("[error] --cache-hit-rate 需在 [0, 1) 之间")

    # 解析链桶: {input_len: [stage_lengths]}
    use_chain = not args.no_chain
    chain_bucket_map = {}
    if use_chain:
        requested = [int(s.strip()) for s in args.chain_buckets.split(",") if s.strip()]
        if not requested:
            raise SystemExit("[error] --chain-buckets 不能为空 (或用 --no-chain 禁用链)")
        const_map = {b: s for b, s in DEFAULT_CHAIN_BUCKETS}
        dist_lens = {L for L, _, _ in DISTRIBUTION}
        for b in requested:
            if b not in dist_lens:
                raise SystemExit("[error] --chain-buckets %d 不在 DISTRIBUTION 中" % b)
            if b in const_map:
                stages = list(const_map[b])
            else:
                n = args.chain_num_stages
                step = b / n
                stages = [int(step * (i + 1)) for i in range(n)]
                stages[-1] = b
            for i in range(1, len(stages)):
                if stages[i] <= stages[i - 1]:
                    raise SystemExit("[error] 桶 %d 阶段非递增: %s" % (b, stages))
            if stages[-1] != b:
                raise SystemExit("[error] 桶 %d 最后阶段必须 == %d, 实际 %d" % (b, b, stages[-1]))
            chain_bucket_map[b] = stages
        # 交错排列要求所有链桶阶段数相同
        stage_counts = {len(s) for s in chain_bucket_map.values()}
        if len(stage_counts) > 1:
            raise SystemExit(
                "[error] 所有链桶阶段数必须相同 (当前 %s); 请改 CHAIN_BUCKETS 常量统一阶段数"
                % {b: len(s) for b, s in chain_bucket_map.items()})

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
        print("[info]   链桶 = %s" % list(chain_bucket_map.keys()))
        for b, stages in chain_bucket_map.items():
            print("[info]     桶 %d 阶段 = %s" % (b, stages))
        print("[info]   中间阶段输出 = %d tokens (priming)" % args.chain_intermediate_output)
        print("[info]   最少链数/桶 = %d (--min-chains)" % args.min_chains)
    else:
        print("[info] 渐进式前缀链: DISABLED (--no-chain)")

    counts = allocate_counts(args.num_requests, DISTRIBUTION)
    builder = Builder(tok, questions, args.cache_hit_rate, DISTRIBUTION,
                      args.kv_block_size, args.body_salt, args.seed)

    # ---- 名额分配 + 每个链桶的链数提升 ----
    bucket_chain_count = {}  # {input_len: num_chains}
    if use_chain:
        for i, (L, O, _pct) in enumerate(DISTRIBUTION):
            if L in chain_bucket_map:
                nc = counts[i]
                if nc < args.min_chains:
                    print("[info] %d 桶提升: %d → %d (--min-chains)" % (L, nc, args.min_chains))
                    nc = args.min_chains
                else:
                    print("[info] %d 桶链数 = %d" % (L, nc))
                bucket_chain_count[L] = nc

    out_path = args.output or ("gsm8k_4k_12k_c%d_cache%d_%s.jsonl" % (
        args.concurrency, int(round(args.cache_hit_rate * 100)), run_id))

    # ---- 1. 生成普通请求 (排除所有链桶) ----
    # [C5] 普通请求先 shuffle, 之后再与链阶段交错排列, 交错后不再 shuffle
    normal_requests = []
    rid = 0
    sum_in = 0        # 名义输入 token 总数
    sum_shared = 0    # 共享(可命中)token 总数
    dist_iter = list(zip(DISTRIBUTION, counts))
    for i, ((L, O, _pct), cnt) in enumerate(dist_iter):
        if use_chain and L in chain_bucket_map:
            # 链桶: target stage 替代普通请求, 贡献 sum_in/sum_shared
            for _ in range(bucket_chain_count[L]):
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

    # ---- 2. 生成渐进式前缀链 (多桶, 每桶独立 ProgressiveChainBuilder) ----
    all_chain_groups = []   # 所有链 (每条链 = stage 请求列表), 供交错排列
    chain_builders = {}     # {input_len: ProgressiveChainBuilder}, 供统计与汇总
    if use_chain:
        for L, stages in chain_bucket_map.items():
            cb = ProgressiveChainBuilder(builder, stages, args.kv_block_size)
            chain_builders[L] = cb
            num_chains_L = bucket_chain_count[L]
            target_out_L = next(O for LL, O, _ in DISTRIBUTION if LL == L)
            num_stages = len(stages)
            for chain_idx in range(num_chains_L):
                stage_texts = cb.build_chain(chain_idx)
                chain_requests = []
                for stage_i, stage_txt in enumerate(stage_texts):
                    out_tok = target_out_L if stage_i == num_stages - 1 \
                             else args.chain_intermediate_output
                    chain_requests.append({
                        "question": stage_txt,
                        "answer": "none",
                        "max_tokens": out_tok
                    })
                all_chain_groups.append(chain_requests)

            # 该桶链阶段加入 sum_in/sum_shared 统计
            for _ in range(num_chains_L):
                for stage_i in range(num_stages):
                    sum_in += cb.aligned_stages[stage_i]
                    if stage_i == 0:
                        sum_shared += min(cb.chain_shared_len, cb.aligned_stages[0])
                    else:
                        sum_shared += cb.aligned_stages[stage_i] - \
                                      cb.aligned_stages[stage_i - 1]

    # ---- 3. 交错排列 (所有链桶的链统一交错, 阶段数已校验一致) ----
    if use_chain and all_chain_groups:
        gap = args.chain_gap
        if gap <= 0:
            num_stages = len(all_chain_groups[0])
            total_chains = len(all_chain_groups)
            auto_gap = len(normal_requests) // (num_stages * total_chains + 1)
            gap = max(1, min(args.concurrency, auto_gap))
        print("[info] 链 stage 间隔 = %d 个普通请求" % gap)

        requests = interleave_requests(normal_requests, all_chain_groups, gap)
        num_stages = len(all_chain_groups[0])
        total_chains = len(all_chain_groups)
        print("[info] 交错排列: %d 普通请求 + %d 链×%d 阶段 = %d 总请求" % (
            len(normal_requests), total_chains, num_stages, len(requests)))
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
        if use_chain and L in chain_bucket_map:
            actual_cnt = bucket_chain_count[L]
            share = 100.0 * builder.shared_len(L) / L
            print("%-14s %-8.2f %-8d %-8.1f %s" % (
                "%d/%d" % (L, O), pct, actual_cnt, share, "链×%d" % actual_cnt))
            continue
        if cnt == 0:
            print("%-14s %-8.2f %-8d %-8s %s" % ("%d/%d" % (L, O), pct, 0, "-", "-"))
            continue
        sample_q, _ = builder.make_question(L, idx)
        act = len(tok.encode(sample_q))
        idx += cnt
        share = 100.0 * builder.shared_len(L) / L
        print("%-14s %-8.2f %-8d %-8.1f %d" % (
            "%d/%d" % (L, O), pct, cnt, share, act))

    achieved = 100.0 * (sum_shared - max(1, builder.shared_len(max(L for L, _, _ in DISTRIBUTION)))) / max(sum_in, 1)
    print("-" * 60)
    print("[done] 写出 %d 条 -> %s" % (len(requests), os.path.abspath(out_path)))
    print("[info] 名义输入 token 合计 = %d, 共享前缀 token 合计 = %d" % (sum_in, sum_shared))
    print("[info] 共享前缀语料偏移 = %d" % builder.shared_offset)
    print("[info] 批次标记 token 数 = %d（共享区每个块均带批次标记）" % len(builder.batch_marker_ids))
    print("[info] 正文唯一化块大小 = %d" % builder.unique_block_size)
    print("[info] 目标 cache 命中率 = %.1f%%, 估算聚合命中率 ≈ %.1f%% (扣除首次写入)" % (
        100.0 * args.cache_hit_rate, achieved))
    if use_chain and all_chain_groups:
        num_stages = len(all_chain_groups[0])
        total_chains = len(all_chain_groups)
        total_priming = total_chains * (num_stages - 1)
        print("[info] 渐进链: %d 链 × %d 阶段 = %d 链请求 (含 %d priming)" % (
            total_chains, num_stages, total_chains * num_stages, total_priming))
        for L in chain_bucket_map:
            cb = chain_builders[L]
            print("[info]   桶 %d: shared_len=%d (%.0f%% of %d), 阶段对齐=%s" % (
                L, cb.chain_shared_len, 100.0 * args.cache_hit_rate,
                cb.target_len, cb.aligned_stages))
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
