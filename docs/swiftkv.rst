
.. _swiftkv:

=======
SwiftKV
=======

SwiftKV is an inference optimization technique designed to reduce compute
overhead during the prefill phase of large language model (LLM) inference,
particularly when processing long input prompts. It introduces a method called
SingleInputKV, which allows later transformer layers to reuse key-value (KV)
pairs computed by earlier layers, eliminating redundant computation.

This technique improves throughput and reduces latency without modifying model
weights. In benchmarks with models like Llama 3.1 70B, SwiftKV reduced prefill
computation by up to 50%, offering a practical performance gain for serving LLMs
efficiently.

You can read more about SwiftKV in the `Snowflake blog post
<https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/>`_
and the `arXiv paper <https://arxiv.org/abs/2410.03960>`_.

---------------------------
Usage with Arctic Inference
---------------------------

To use SwiftKV with Arctic Inference, you need to :ref:`install <install>` the
``arctic-inference`` package and select a SwiftKV model that has been fine-tuned
with `SwiftKV in ArcticTraining
<https://github.com/snowflakedb/ArcticTraining/tree/main/projects/swiftkv>`_. We
have publically released SwiftKV models for Meta's Llama-3 series of models `on
Hugging Face
<https://huggingface.co/collections/Snowflake/swiftkv-models-674f7d7474eb789e185d31cb>`_.

Loading one of these models will automatically enable the SwiftKV optimization.
For example, to load the `meta-llama/Llama-3.3-70B-Instruct
<https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ model with SwiftKV,
you would select the `Snowflake/Llama-3.3-SwiftKV-70B-Instruct
<https://huggingface.co/Snowflake/Llama-3.3-SwiftKV-70B-Instruct>`_ model:

.. code-block:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model Snowflake/Llama-3.3-SwiftKV-70B-Instruct \
       --tensor-parallel-size 8


----------------------------------
Training SwiftKV-Compatible Models
----------------------------------

If your favorite model is not already available as a `SwiftKV-compatible model
<https://huggingface.co/collections/Snowflake/swiftkv-models-674f7d7474eb789e185d31cb>`_,
you can fine-tune it with `ArcticTraining
<https://github.com/snowflakedb/ArcticTraining>`_ to make it compatible with
SwiftKV. 

To get started, refer to our provided `examples
<https://github.com/snowflakedb/ArcticTraining/tree/main/projects/swiftkv/configs>`_
for how we fine-tuned the Llama-3 and Qwen-2.5 models with SwiftKV