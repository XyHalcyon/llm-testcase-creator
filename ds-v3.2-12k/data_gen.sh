python3 gen_data_12k.py \
  --gsm8k-path "./GSM8K.jsonl" \
  --tokenizer "/apps/models/DeepSeek-V3.2-w8a8-mtp-QuaRot/" \
  --num-requests 2620 \
  --cache-hit-rate 0.25 \
  --kv-block-size 128 \
  --concurrency 262 \
  --format evalscope
