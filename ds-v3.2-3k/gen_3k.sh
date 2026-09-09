python gen_data_3k_unique.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V3.2-w8a8-mtp-QuaRot/" \
  --num-requests 2870 \
  --cache-hit-rate 0.15 \
  --kv-block-size 128 \
  --concurrency 287 \
  --format evalscope
