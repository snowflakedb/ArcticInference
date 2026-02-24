#!/usr/bin/env bash
# HTTP API example for arctic_inference.server
#
# Start the server first:
#   arctic_inference_server --port 8000
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

# Helper to run a generation request and report timing/token stats
run_request() {
  local id="$1"
  local prompt="$2"
  local start_time end_time elapsed total_tokens
  start_time=$(date +%s.%N)
  response=$(curl -s -X POST "$BASE_URL/sample" \
    -H "Content-Type: application/json" \
    -d "{
      \"prompts\": [\"$prompt\"],
      \"sampling_params\": {\"max_tokens\": 2048, \"temperature\": 0.7}
    }")
  end_time=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")
  total_tokens=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(sum(len(x['token_ids']) for x in r['results']))" 2>/dev/null || echo "error")
  echo "[Request $id] time=${elapsed}s tokens=$total_tokens"
}

echo "=== 1. Initialize model ==="
curl -s -X POST "$BASE_URL/init" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "tensor_parallel_size": 1,
    "quantization": "fp8",
    "max_model_len": 4096
  }' | python3 -m json.tool

echo ""
echo "=== 2. Check status ==="
curl -s "$BASE_URL/status" | python3 -m json.tool

echo ""
echo "=== 3. Parallel generation test ==="
prompts=(
  "What is the capital of France?"
  "Explain quantum computing in one sentence."
  "What is the meaning of life?"
  "Write a haiku about programming."
  "What is the speed of light?"
  "Describe photosynthesis briefly."
  "Name three prime numbers."
  "What is machine learning?"
  "Who wrote Romeo and Juliet?"
  "What causes rainbows?"
  "Define entropy in physics."
  "What is the Pythagorean theorem?"
  "Explain recursion simply."
  "What is DNA?"
  "Name three oceans."
  "What is gravity?"
)

overall_start=$(date +%s.%N)
for i in "${!prompts[@]}"; do
  run_request "$((i + 1))" "${prompts[$i]}" &
done
wait
overall_end=$(date +%s.%N)
overall_elapsed=$(awk "BEGIN {printf \"%.3f\", $overall_end - $overall_start}")
echo "All ${#prompts[@]} requests completed in ${overall_elapsed}s"

echo ""
echo "=== 4. Shutdown ==="
curl -s -X POST "$BASE_URL/shutdown" | python3 -m json.tool
