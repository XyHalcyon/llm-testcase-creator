# 生成 DeepSeek V4 Flash 12K 数据集 (含 42100+84200 双桶渐进式前缀链)
# 哪些桶做链由 gen_data_4k_12k.py 顶部 CHAIN_BUCKETS 常量控制, 也可用 --chain-buckets 覆盖
python gen_data_4k_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V4-Flash/" \
  --num-requests 3660 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 128 \
  --min-chains 3 \
  --chain-gap 0 \
  --format evalscope \
  --rate 4
