# ArcticInference

ArcticInference contains optimizations developed and used at Snowflake that
improve the performance of serving LLMs. ArcticInference can be installed as a
vLLM plugin and run with existing vLLM v1 installations.

## Installation

```console
$ git clone https://github.com/snowflakedb/ArcticInference.git
$ cd ArcticInference && pip install .[vllm]
```

## SwiftKV on vLLM

SwiftKV is a technique developed by Snowflake AI Research that reduces computational overhead during prompt processing by combining model rewiring and knowledge-preserving self-distillation.

For more details, see:

- [Blog post](https://www.snowflake.com/engineering-blog/swiftkv-llm-compute-reduction)
- [Paper](https://arxiv.org/abs/2410.03960)
- [Huggingface](https://huggingface.co/collections/Snowflake/swiftkv-models-674f7d7474eb789e185d31cb)

### Running SwiftKV

Run an example conversation using [Snowflake/Llama-3.1-SwiftKV-8B-Instruct](https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct):
```console
$ python examples/offline_inference_swiftkv.py

...

The Importance of Higher Education

Higher education is a vital component of an individual's life, providing numerous benefits that extend beyond the acquisition of knowledge and skills. It plays a significant role in shaping an individual's future, career prospects, and overall well-being. In this essay, we will explore the importance of higher education and its far-reaching implications on individuals, society, and the economy.

...
```
