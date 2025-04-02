# ArcticInference

ArcticInference is a new library from Snowflake AI Research that contains current and future LLM inference optimizations developed at Snowflake. It is integrated with vLLM v0.8.1 using vLLM’s custom plugin feature, allowing us to develop and integrate inference optimizations quickly into vLLM and make them available to the community. 

Once installed, ArcticInference automatically patches vLLM to use Arctic Ulysses and other optimizations implemented in ArcticInference, and users can continue to use their familiar vLLM APIs and CLI. It’s easy to get started!

## Installation

```console
$ git clone https://github.com/snowflakedb/ArcticInference.git
$ cd ArcticInference && pip install .[vllm]
```

## Projects 
To better understand what features ArcticInference supports please refer to the following list of projects we have released under this framework:

* [SwiftKV](projects/swiftkv)
* [Arctic Ulysses](projects/ulysses)
