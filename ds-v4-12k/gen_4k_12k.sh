# 生成 DeepSeek V4 Flash 混合序列数据集 (双文件模式: 预热文件 + 正式文件)
# 预热文件: 各预热桶的前缀请求 (长度 < 正式长度), 先跑它把前缀写入服务端 prefix cache
# 正式文件: 纯普通混合序列 + 预热桶完整长度请求 (与预热请求 token 级前缀一致, 可命中缓存)
/usr/local/uv/envs/llmcase/bin/python gen_data_4k_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/workspace/llm-testcase-creator/ds-v4-12k/deepseek-v4-flash-tokenizer" \
  --num-requests 2160 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --min-pairs 3 \
  --warmup-output "./warmup_prefix.jsonl" \
  --format evalscope \
  --rate 4
