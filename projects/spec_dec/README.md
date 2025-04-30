## Arctic Speculator on vLLM

Arctic Speculator is a comprehensive Speculative Decoding strategy that combines [Suffix Decoding](https://arxiv.org/pdf/2411.04975), easy-to-use speculator training recipes through [ArcticTraining](https://github.com/snowflakedb/ArcticTraining/tree/main) and optimized speculative inference pipeline through ArcticInference. With this strategy on top of vLLM V1, we achieve up to 4x faster end-to-end task completion for LLM agents and up to 2.8x faster decoding for open-ended interactive workloads compared with vLLM without speculation.

<img src="Arctic Speculator Benchmarking.png" alt="" width="900">

\
For more details, see:
- [Blog Post: Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)

### Running Arctic Speculator

Run an example conversion using [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as base model and [Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct](https://huggingface.co/Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct) as draft model:
```console
$ python offline_inference_spec_dec.py
