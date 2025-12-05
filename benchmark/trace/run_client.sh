
model="/data-fast/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
dataset=/home/yak/toolagent_trace.jsonl

vllm bench serve --model $model --trace-dataset-path $dataset

