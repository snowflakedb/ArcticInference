
model="/data-fast/RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
dataset=example_trace.jsonl

vllm bench serve --model $model --trace-dataset-path $dataset --ignore-eos

