#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_data_3k.py — 生成 AISBench 性能测试用的 GSM8K 混合序列数据集。

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

运行 (示例):
  python gen_data_3k.py                       # 默认 2500 条, 32 并发, 15% cache
  python gen_data_3k.py --gsm8k-path D:/download/xxx.jsonl --tokenizer /path/to/deepseek-v3.2
  然后:
  ais_bench --models <model> \
            --custom-dataset-path gsm8k_3k_c32_cache15.jsonl \
            --custom-dataset-data-type qa \
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


# ------------------------- 内容构造 -------------------------
def tile_to(ids, n):
    """把 ids 重复/截断到长度 n。"""
    if len(ids) >= n:
        return ids[:n]
    out = []
    while len(out) < n:
        out.extend(ids[:n - len(out)])
    return out


class Builder:
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


# ------------------------- EvalScope 适配 -------------------------
# 每行是一个完整 Chat 请求 dict(带 messages + 逐请求 max_tokens + ignore_eos)。
# EvalScope build_request 的 dict 分支用 preserve_existing=True(setdefault),
# 全局 --max-tokens 不会覆盖行内 max_tokens -> 单次 run 内混合序列+各自输出长度。
_EVALSCOPE_PLUGIN = '''# -*- coding: utf-8 -*-
"""gen_data_3k.py 配套的 EvalScope 自定义数据集 plugin(注册 dataset 名 'mixed')。
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

# 一键启动脚本: 关键是 max_tokens=None(否则默认 2048 覆盖每行值)。已在 evalscope 1.9.0 验证。
_EVALSCOPE_RUNNER = '''# -*- coding: utf-8 -*-
"""一键跑 EvalScope perf(混合序列 + 逐请求输出长度)。
改下面 model/url/tokenizer_path 后: python run_perf.py
关键: max_tokens=None —— EvalScope 全局 max_tokens 默认 2048 会覆盖每行的值,
      设 None 才让数据里每行自带的 max_tokens 生效(evalscope 1.9.0 实测通过)。
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
    rate=-1,           # 闭环并发; 想按 QPS 到达改成 rate=<req/s> 并调大 parallel
    max_tokens=None,   # 别改: None 才逐请求生效
    stream=True,
    name="mix_perf",
)
run_perf_benchmark(args)
'''


def write_evalscope(requests, out_path, concurrency):
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
                                     "parallel": concurrency})
    return plugin_path, runner_path


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="生成 AISBench GSM8K 混合序列 + 15% cache 命中率性能数据集")
    ap.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                    help="总请求数 (默认 %d, 保证最稀有的桶>=1条)" % DEFAULT_NUM_REQUESTS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="并发数, 仅用于文件名与运行提示 (默认 %d)" % DEFAULT_CONCURRENCY)
    ap.add_argument("--cache-hit-rate", type=float, default=DEFAULT_CACHE_RATE,
                    help="共享前缀占每条 prompt 的比例 (默认 %.2f)" % DEFAULT_CACHE_RATE)
    ap.add_argument("--gsm8k-path", default=None,
                    help="本地 GSM8K jsonl (每行含 question); 不给则用 HF datasets")
    ap.add_argument("--tokenizer", default=None,
                    help="transformers 分词器名/路径 (给了才用, 如 DeepSeek-V3.2 权重目录)")
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
    args = ap.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(4)
    if not args.body_salt:
        args.body_salt = run_id

    if not (0.0 <= args.cache_hit_rate < 1.0):
        raise SystemExit("[error] --cache-hit-rate 需在 [0, 1) 之间")

    random.seed(args.seed)

    tok, tok_desc = build_tokenizer(args.tokenizer)
    questions = load_questions(args.gsm8k_path)
    print("[info] tokenizer = %s" % tok_desc)
    print("[info] GSM8K 问题数 = %d" % len(questions))
    print("[info] 本次数据集盐值 = %s" % args.body_salt)

    if args.kv_block_size > 0:
        print("[info] 块对齐模式: KV block=%d (共享前缀取整到整块)" % args.kv_block_size)

    counts = allocate_counts(args.num_requests, DISTRIBUTION)
    builder = Builder(tok, questions, args.cache_hit_rate, DISTRIBUTION,
                      args.kv_block_size, args.body_salt, args.seed)

    out_path = args.output or ("gsm8k_3k_c%d_cache%d_%s.jsonl" % (
        args.concurrency, int(round(args.cache_hit_rate * 100)), run_id))

    requests = []
    rid = 0
    sum_in = 0        # 名义输入 token 总数
    sum_shared = 0    # 共享(可命中)token 总数
    for (L, O, _pct), cnt in zip(DISTRIBUTION, counts):
        for _ in range(cnt):
            q, slen = builder.make_question(L, rid)
            requests.append({"question": q, "answer": "none", "max_tokens": O})
            sum_in += L
            sum_shared += slen
            rid += 1
    random.shuffle(requests)

    plugin_path = runner_path = None
    if args.format == "evalscope":
        plugin_path, runner_path = write_evalscope(requests, out_path, args.concurrency)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------- 汇总 --------
    print("\n%-14s %-8s %-8s %-8s %s" % ("in/out", "pct%", "count", "share%", "act.in-tok"))
    print("-" * 60)
    # 抽样校验实际分词长度漂移 (只测每桶第一条, 便宜)
    idx = 0
    for (L, O, pct), cnt in zip(DISTRIBUTION, counts):
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

    achieved = 100.0 * (sum_shared - max(1, builder.shared_len(70000))) / max(sum_in, 1)
    print("-" * 60)
    print("[done] 写出 %d 条 -> %s" % (len(requests), os.path.abspath(out_path)))
    print("[info] 名义输入 token 合计 = %d, 共享前缀 token 合计 = %d" % (sum_in, sum_shared))
    print("[info] 共享前缀语料偏移 = %d" % builder.shared_offset)
    print("[info] 批次标记 token 数 = %d（位于共享前缀最前面）" % len(builder.batch_marker_ids))
    print("[info] 目标 cache 命中率 = %.1f%%, 估算聚合命中率 ≈ %.1f%% (扣除首次写入)" % (
        100.0 * args.cache_hit_rate, achieved))
    if args.format == "evalscope":
        print("[info] EvalScope plugin -> %s" % plugin_path)
        print("[info] EvalScope runner -> %s" % runner_path)
        print("\n[run] 改 run_perf.py 里的 model/url/tokenizer_path 后直接跑:")
        print("        python %s" % os.path.basename(runner_path))
        print("  # 关键: 用 chat/completions 端点 + max_tokens=None(runner 已内置),")
        print("  #       否则 EvalScope 全局 max_tokens(默认2048)会覆盖每行的输出长度。")
    else:
        print("\n[run] AISBench 示例命令:")
        print("  ais_bench --models <your_model> \\")
        print("            --custom-dataset-path %s \\" % out_path)
        print("            --custom-dataset-data-type qa \\")
        print("            --max-out-len -1        # 使用每条的 max_tokens")
        print("  # 并发在 AISBench 运行期设置 (如 --batch-size/并发相关参数 = %d)" % args.concurrency)


if __name__ == "__main__":
    main()
