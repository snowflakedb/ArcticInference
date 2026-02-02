
model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
vllm bench serve --model $model  --trace-dataset-path AzureLLMInferenceTrace_code_15mins.jsonl --ignore-eos --trace-output-path code_output.csv

# model="Qwen/Qwen3-32B-FP8"
# vllm bench serve --model $model --trace-dataset-path conversation_trace_15mins.jsonl --ignore-eos --trace-output-path conversation_output.csv
