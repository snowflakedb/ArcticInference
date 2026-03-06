#!/usr/bin/env bash
# Simple single-model HTTP API example for arctic_inference.server
#
# Start the server first:
#   arctic_inference_server --port 8000
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== 1. Init model ==="
curl -s -X POST "$BASE_URL/init" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model": "Qwen/Qwen3-30B-A3B",
      "tensor_parallel_size": 1,
      "quantization": "fp8",
      "max_model_len": 4096
    },
    "model_id": "qwen"
  }' | python3 -m json.tool

echo ""
echo "=== 2. Status ==="
curl -s "$BASE_URL/status" | python3 -m json.tool

echo ""
echo "=== 3. Generate ==="
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen",
    "prompts": [
      "What is the capital of France?",
      "Explain quantum computing in one sentence.",
      "Name three prime numbers.",
      "What is the speed of light?"
    ],
    "sampling_params": {"max_tokens": 256, "temperature": 0.7}
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, r in enumerate(data['results']):
    tokens = len(r['token_ids'])
    print(f'  [#{i+1}] {tokens} tokens: {r[\"text\"][:80]}...')
"

echo ""
echo "=== 4. Shutdown ==="
curl -s -X POST "$BASE_URL/shutdown" | python3 -m json.tool
