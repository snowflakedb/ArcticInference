[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/snowflakedb/ArcticInference/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/arctic-inference.svg)](https://pypi.org/project/arctic-inference/)

## Latest news
* [2025/05] - [Arctic Inference with Shift Parallelism: The Fastest Open Source Inference System for Enterprise AI](https://www.snowflake.com/en/engineering-blog/arctic-inference-shift-parallelism/)
* [2025/05] - [Scaling vLLM for Embeddings: 16x Throughput and Cost Reduction](https://www.snowflake.com/en/engineering-blog/embedding-inference-arctic-16x-faster/)
* [2025/05] - [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
* [2025/04] - [Low-Latency and High-Throughput Inference for Long Context with Sequence Parallelism (aka Ulysses)](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)

# Arctic Inference
<img src="projects/arctic_inference/imgs/figure1.png" alt="" width="900">

\
Inference is becoming the dominant workload in AI, but today’s systems force developers to make costly trade-offs between low latency, high throughput and affordable deployment. 

Arctic Inference changes that. Built by Snowflake AI Research, it’s an open source vLLM plugin that brings Snowflake’s inference innovations to the community, delivering the fastest, most cost-effective open source inference for enterprise AI. 

For real-world generative AI workloads, Arctic Inference + vLLM in a single deployment, achieves:

- 3.4x faster request completion and 1.06x higher throughput compared to SoTA throughput-optimized deployment (TP=1, DP=8)
- 1.7x higher throughput and 1.28x faster request completion compared to SoTA latency-optimized deployment (TP=8, DP=1) 


<p align="middle">
<img src="projects/arctic_inference/imgs/trifecta.png" alt="" width="322">
<img src="projects/arctic_inference/imgs/embedding.png" alt="" width="410">
</p>

Arctic Inference + vLLM also achieves the elusive trifecta: Quicker response, higher throughput and faster generation:
- 2.25x faster response time (prefill throughput per request) 
- 1.75x faster generation per request
- on-par combined throughput

For non-generative AI workloads, such as embeddings, Arctic Inference + vLLM delivers a whopping 1.4M toks/sec per GPU:
- 16x faster than vLLM on short sequences and 4.2x faster on long sequences
- 2.4x faster than Text Embeddings Inference (TEI) on short sequences and at parity for long sequences

\
The performance claims are supported with detailed evaluation results. To learn more check out our blog [Arctic Inference with Shift Parallelism: The Fastest Open Source Inference System for Enterprise AI](https://www.snowflake.com/en/engineering-blog/arctic-inference-shift-parallelism/)

## Installation

```console
$ pip install arctic-inference[vllm]
```
Once installed, Arctic Inference automatically patches vLLM to use Arctic Inference with Shift Parallelism and other optimizations implemented in Arctic Inference, and users can continue to use their familiar vLLM APIs and CLI. It’s easy to get started!

## Projects 
To better understand what features Arctic Inference supports please refer to the following list of projects we have released under this framework:

* Shift Parallelism
* [Arctic Ulysses](projects/ulysses)
* [Arctic Speculator](projects/spec_dec/)
* [SwiftKV](projects/swiftkv)
* [Arctic Embedding](arctic_inference/embedding/)


## Running Arctic Inference for Generation with all the features
By using the examples below, you are expected to get benefits from Shift Parallelism, Speculative Decoding and SwiftKV all at once!
### Serving
```console
vllm serve \
Snowflake/Llama-3.1-SwiftKV-70B-Instruct \
--quantization "fp8" \
--tensor-parallel-size 1 \
--ulysses-sequence-parallel-size 4 \
--enable-shift-parallel \
--speculative-config '{
    "method": "arctic",
    "model":"Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct",
    "num_speculative_tokens": 3,
    "enable_suffix_decoding": true,
    "disable_by_batch_size": 64,
}'

```
### Offline
```python
import vllm
from vllm import LLM, SamplingParams

vllm.plugins.load_general_plugins()

llm = LLM(
    model="Snowflake/Llama-3.1-SwiftKV-70B-Instruct",
    quantization="fp8",
    tensor_parallel_size=1,
    ulysses_sequence_parallel_size=4,
    enable_shift_parallel=True,
    speculative_config={
        "method": "arctic",
        "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct",
        "num_speculative_tokens": 3,
        "enable_suffix_decoding": True,
        "disable_by_batch_size": 64,
    },
)

conversation = [
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=128)

outputs = llm.chat(conversation, sampling_params=sampling_params)
```


