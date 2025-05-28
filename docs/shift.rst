
.. _shift:

===================
Shift Parallelism
===================

Shift Parallelism is a dynamic inference parallelism strategy that adapts
between tensor parallelism (TP) and Arctic sequence parallelism (SP) in real
time, optimizing for latency, throughput, and cost efficiency — all within a
single deployment. Instead of statically tuning for one metric, Shift
Parallelism responds to real-world traffic by switching modes based on batch
size: using TP for small batches (minimizing output token latency), and SP for
large batches (maximizing throughput and minimizing time-to-first-token).

This seamless switching is enabled by KV cache invariance — the cache layout
remains consistent between TP and SP as long as `TP x SP = P` (total
parallelism), allowing the system to transition modes without disruption.

For more details, refer to the `Snowflake blog post
<https://www.snowflake.com/en/engineering-blog/INSERT-LINK>`_.

--------------------------
Usage with Arctic Inference
--------------------------

To use Shift Parallelism with Arctic Inference, :ref:`install <install>` the
``arctic-inference`` package, select a compatible `Llama-3
<https://huggingface.co/models?other=llama-3>`_ model and launch vLLM with a
tensor and sequence parallel configuration where `TP x SP` equals the number of
GPUs.

Arctic Inference will automatically detect traffic conditions and activate the
most optimal mode at runtime.

Here is an example of how run Shift Parallelism with the
`meta-llama/Llama-3.3-70B-Instruct
<https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ model:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --enable-shift-parallel \
        --tensor-parallel-size 4 \
        --ulysses-sequence-parallel-size 2 \
        --shift-parallel-threshold 256

This enables Arctic Inference to dynamically balance latency and throughput
without manual intervention or separate deployments.
