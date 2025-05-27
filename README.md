[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/snowflakedb/ArcticInference/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/arctic-inference.svg)](https://pypi.org/project/arctic-inference/)

## Latest news
* [2025/05] - [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
* [2025/04] - [Low-Latency and High-Throughput Inference for Long Context with Sequence Parallelism (aka Ulysses)](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)

# ArcticInference

ArcticInference is a new library from Snowflake AI Research that contains current and future LLM inference optimizations developed at Snowflake. It is integrated with vLLM v0.8.1 using vLLM’s custom plugin feature, allowing us to develop and integrate inference optimizations quickly into vLLM and make them available to the community. 

Once installed, ArcticInference automatically patches vLLM to use Arctic Ulysses and other optimizations implemented in ArcticInference, and users can continue to use their familiar vLLM APIs and CLI. It’s easy to get started!

## Installation

```console
$ pip install arctic-inference[vllm]
```

## Projects 
To better understand what features ArcticInference supports please refer to the following list of projects we have released under this framework:

* [SwiftKV](projects/swiftkv)
* [Arctic Ulysses](projects/ulysses)
* [Arctic Speculator](projects/spec_dec/)

## Running ArcticInference with all the features
### Serving
```console
vllm serve \
Snowflake/Llama-3.1-SwiftKV-70B-Instruct \
--quantization "fp8" \
--shift-parallel-min-tp-size 1 \
--shift-parallel-max-tp-size 4 \
--shift-parallel-threshold 512 \
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
    shift_parallel_threshold=512,
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
