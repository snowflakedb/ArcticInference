
.. _ulysses:

==============
Arctic Ulysses
==============

Arctic Ulysses is a sequence parallelism technique designed to improve inference
performance for large language models (LLMs) on long-context inputs. Unlike
traditional tensor parallelism (TP), which partitions model computation across
GPUs and incurs significant inter-GPU communication overhead, Arctic Ulysses
partitions the input sequence itself. This approach reduces time-to-first-token
(TTFT) latency and enhances throughput efficiency, particularly for tasks like
retrieval-augmented generation (RAG), summarization, and code generation.

By leveraging all-to-all communication for attention computation, Arctic Ulysses
minimizes communication overhead and maintains a favorable
communication-to-compute ratio as the number of GPUs scales. In evaluations,
Arctic Ulysses achieved up to 6.8× lower latency and 1.5× higher throughput
compared to TP-only configurations, without requiring multiple specialized
deployments.

For more details, refer to the `Snowflake blog post
<https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/>`_.

--------------------------
Usage with ArcticInference
--------------------------

To utilize Arctic Ulysses with ArcticInference, :ref:`install <install>` the
``arctic-inference`` package and select a compatible `Llama-3
<https://huggingface.co/models?other=llama-3>`_ or or `Qwen-2
<https://huggingface.co/models?other=qwen2>`_ model.

.. note::

    Currently, Arctic Ulysses works only with certain model architectures,
    namely Llama and Qwen. We are working on enabling Ulysses more broadly
    across any model architecture, which will be part of a later release of
    ArcticInference.

Then when launching vLLM, specifying both ``tensor-parallel-size`` and
``sequence-parallel-size`` will automatically enable the Arctic Ulysses
optimization.  Here's an example of how to run the
`meta-llama/Llama-3.3-70B-Instruct
<https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ model with both
tensor and sequence parallelism across 8 GPUs (4 TP, 2 SP) with ArcticInference:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 4 \
        --sequence-parallel-size 2
