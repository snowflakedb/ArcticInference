.. _spec-decode:

====================
Speculative Decoding
====================

Speculative Decoding is an inference acceleration technique that leverages a
smaller, faster "draft" model to propose multiple output tokens, which are then
verified by the larger target model. This approach enables parallel token
generation, significantly reducing decoding latency without altering the output
distribution.

In benchmarks, combining Speculative Decoding with Arctic Inference and vLLM has
achieved up to 4× faster end-to-end task completion for LLM agents and up to
2.8× faster decoding for conversational, interactive, and coding workloads.

For more details, refer to the `Snowflake blog post
<https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/>`_.

--------------------------
Usage with ArcticInference
--------------------------

To utilize Speculative Decoding with ArcticInference, :ref:`install <install>`
the ``arctic-inference`` package and select a model pair comprising a draft
model and a target model. The draft model should be trained to closely
approximate the output distribution of the target model with `MLP Speculator in
ArcticTraining
<https://github.com/snowflakedb/ArcticTraining/tree/main/projects/mlp_speculator>`_.
We have publically released draft models for the Llama-3 and Qwen-2.5 series of
models `on Hugging Face
<https://huggingface.co/collections/Snowflake/speculators-6812b07f3186d13e243022e4>`_.`

Specifying a ``speculative-config`` with ``method="arctic"`` will enable
Speculative Decoding via ArcticInference when launching vLLM. For example, to
load `meta-llama/Llama-3.3-70B-Instruct
<https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ with the
`Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct
<https://huggingface.co/Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct>`_
draft model:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --quantization "fp8" \
        --tensor-parallel-size 2 \
        --speculative-config '{
            "method": "arctic",
            "model":"Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": true
        }'

-----------------------------------------
Training Draft Models with ArcticTraining
-----------------------------------------

If a suitable draft model is in `our publically shared draft model list
<https://huggingface.co/collections/Snowflake/speculators-6812b07f3186d13e243022e4>`_, you can train one using
`ArcticTraining <https://github.com/snowflakedb/ArcticTraining>`_.
ArcticTraining supports the knowledge distillation process required to train a
draft model that closely matches the target model's output distribution.

To get started, refer to our provided `Llama-3.1-8B-Instruct example
<https://github.com/snowflakedb/ArcticTraining/blob/main/projects/mlp_speculator/llama-8b.yaml>`_
and adapt it to your training needs.
