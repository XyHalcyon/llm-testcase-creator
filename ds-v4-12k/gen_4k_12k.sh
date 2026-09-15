# 生成 DeepSeek V4 Flash 混合序列数据集 (含 344158+499835+728183 三桶渐进式前缀链)
# 哪些桶做链由 gen_data_4k_12k.py 顶部 CHAIN_BUCKETS 常量控制, 也可用 --chain-buckets 覆盖
python gen_data_4k_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/workspace/llm-testcase-creator/ds-v4-12k/deepseek-v4-flash-tokenizer" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --min-chains 3 \
  --chain-gap 128 \
  --format evalscope \
  --rate 4
