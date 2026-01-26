
export VLLM_DISABLE_COMPILE_CACHE=1

# two models, three parallelisms, six experiments

model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
ARCTIC_INFERENCE_ENABLED=1 vllm serve $model --disable-log-requests --no-enable-prefix-caching --ulysses-sequence-parallel-size 8 --enable-shift-parallel --max-num-batched-tokens 131072
# vllm serve $model --disable-log-requests --no-enable-prefix-caching --tensor-parallel-size 8 --max-num-batched-tokens 131072
# vllm serve $model --disable-log-requests --no-enable-prefix-caching --data-parallel-size 8 --max-num-batched-tokens 131072

# model="Qwen/Qwen3-32B-FP8"
# ARCTIC_INFERENCE_ENABLED=1 vllm serve $model --disable-log-requests --no-enable-prefix-caching --ulysses-sequence-parallel-size 8 --enable-shift-parallel --max-num-batched-tokens 131072 --kv-cache-dtype fp8
# vllm serve $model --disable-log-requests --no-enable-prefix-caching --tensor-parallel-size 8 --max-num-batched-tokens 131072 --kv-cache-dtype fp8
# vllm serve $model --disable-log-requests --no-enable-prefix-caching --data-parallel-size 8 --max-num-batched-tokens 131072 --kv-cache-dtype fp8
