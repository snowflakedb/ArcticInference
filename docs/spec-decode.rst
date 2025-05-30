
.. _spec-decode:

=============================
Speculative & Suffix Decoding
=============================

Arctic Inference employs advanced techniques like Speculative Decoding and Suffix
Decoding to significantly accelerate LLM inference. It enhances standard
speculative decoding and uniquely integrates it with Suffix Decoding for optimal
performance, reducing latency and improving throughput without altering the
model's output distribution.

Key advantages of Arctic Inference's approach include:

* **Superior Draft Models:** Arctic Inference leverages specially trained draft
  models (MLP/LSTM speculators via ArcticTraining) that achieve high acceptance
  rates, making its speculative decoding component highly efficient.
* **Integrated Suffix Decoding:** Arctic Inference can combine its advanced
  speculative decoding with Suffix Decoding. This synergy allows the system to
  benefit from both general-purpose short-sequence speculation and specialized
  long-sequence speculation for repetitive text.

In benchmarks, combining these techniques with Arctic Inference and vLLM has
achieved up to 4× faster end-to-end task completion for LLM agents and up to
2.8× faster decoding for interactive workloads compared to standard
autoregressive decoding. Arctic Inference's implementation has also shown to be
up to 1.8x faster than other open-source speculative decoding alternatives in
vLLM for certain workloads.

For more in-depth details, refer to the `Snowflake blog post
<https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/>`_.

----------------------------
Understanding the Techniques
----------------------------

Speculative Decoding
********************

Speculative Decoding is an inference acceleration technique that uses a smaller,
faster "draft" model (e.g., an MLP speculator) to propose multiple candidate
output tokens. These proposed tokens are then efficiently verified in parallel
by the larger, more powerful target model. If the proposals are correct,
multiple tokens are accepted at once, speeding up the generation process. The
effectiveness of speculative decoding heavily relies on the quality and
acceptance rate of the draft model's predictions.

Suffix Decoding
***************

Suffix Decoding is a complementary technique particularly effective for text
with repetitive structures, common in agentic workflows. Instead of predicting
a fixed number of tokens, suffix decoding dynamically identifies and speculates
longer sequences by matching patterns from previously generated text (historical
outputs) and the current input. It utilizes a suffix tree data structure to
maintain a cache of sequences, enabling rapid speculation of these recurring
patterns.

---------------------------
Usage with Arctic Inference
---------------------------

To utilize these acceleration techniques with Arctic Inference in vLLM:

1. :ref:`Install <install>` the ``arctic-inference`` package.
2. Select a target model and a corresponding pre-trained draft model.
   Arctic Inference provides public draft models for popular series like Llama-3 and
   Qwen-2.5 `on Hugging Face
   <https://huggingface.co/collections/Snowflake/speculators-6812b07f3186d13e243022e4>`_.

When launching vLLM, specify a ``speculative-config``:

* Set ``"method": "arctic"`` to enable Speculative Decoding via
  Arctic Inference's advanced speculators.
* Optionally, set ``"enable_suffix_decoding": true`` to activate Suffix Decoding
  in conjunction with the speculative method. This is highly recommended for
  workloads with potential textual repetition.

**Example:**

To load `meta-llama/Llama-3.3-70B-Instruct
<https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ with the
`Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct
<https://huggingface.co/Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct>`_
draft model, using both Arctic's speculative decoding and enabling Suffix
Decoding:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --quantization "fp8" \
        --tensor-parallel-size 2 \
        --speculative-config '{
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.3-70B-Instruct",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": true
        }'

This configuration instructs vLLM to use Arctic Inference's specific speculative
decoding logic with the provided draft model and to also leverage Suffix
Decoding for potential further speedups.

------------------------------------------------
Training Custom Draft Models with ArcticTraining
------------------------------------------------

If a pre-trained draft model (speculator) is not available for your target model
in `our public list
<https://huggingface.co/collections/Snowflake/speculators-6812b07f3186d13e243022e4>`_,
you can train your own using `ArcticTraining
<https://github.com/snowflakedb/ArcticTraining>`_. ArcticTraining supports the
knowledge distillation process required to create a high-quality draft model
(e.g., an MLP or LSTM speculator) that closely mimics the target model's output
distribution, which is crucial for effective speculative decoding.

To get started, refer to the `MLP Speculator training examples in ArcticTraining
<https://github.com/snowflakedb/ArcticTraining/tree/main/projects/mlp_speculator>`_,
such as the provided `Llama-3.1-8B-Instruct example
<https://github.com/snowflakedb/ArcticTraining/blob/main/projects/mlp_speculator/llama-8b.yaml>`_,
and adapt it to your specific model and training needs.