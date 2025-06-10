
Arctic Inference Documentation
==============================

Arctic Inference is an open-source vLLM plugin that brings Snowflake's
inference innovations to the community, delivering the fastest and most
cost-effective open-source inference for LLMs and Embeddings.

Once installed, Arctic Inference automatically patches vLLM upon launch to
support the optimizations in Arctic Inference, and users can continue to use
familiar vLLM APIs and CLI. It's easy to get started!

Key Features
------------

Arctic Inference achieves high throughput and low latency through a wholistic
set of inference optimizations.

Advanced Parallelism
~~~~~~~~~~~~~~~~~~~~

🚀 :ref:`shift-parallel` [`blog <https://www.snowflake.com/en/engineering-blog/arctic-inference-shift-parallelism/>`__]:
   Dynamically switches between tensor and sequence parallelism at runtime to
   optimize latency, throughput, and cost — all in one deployment.

🚀 :ref:`arctic-ulysses` [`blog <https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/>`__]:
   Improve long-context inference latency and throughput via sequence
   parallelism across GPUs.

Speculative Decoding
~~~~~~~~~~~~~~~~~~~~

🚀 :ref:`arctic-speculator` [`blog <https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/>`__]:
   Lightweight yet effective draft models based on MLP and LSTM architectures,
   complete with training pipelines.

🚀 :ref:`suffix-decoding` [`paper <https://arxiv.org/pdf/2411.04975>`__, `blog <https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/>`__]:
   Rapid speculation for long repeated sequences, effective for coding, agents
   and other agentic applications.

Model Optimization
~~~~~~~~~~~~~~~~~~

🚀 :ref:`swiftkv` [`paper <https://arxiv.org/pdf/2410.03960>`__]:
   Reduce prefill compute by early-exiting prompt tokens and reusing KV across
   transformer layers.

Other Optimizations
~~~~~~~~~~~~~~~~~~~

🚀 :ref:`embeddings` [`blog <https://www.snowflake.com/en/engineering-blog/embedding-inference-arctic-16x-faster/>`__]:
   Accelerate embedding performance with parallel tokenization, byte outputs,
   and GPU load-balanced replicas.

Getting Started
---------------

Installation
~~~~~~~~~~~~

To install Arctic Inference from PyPI, use the following command:

.. code-block:: bash

   pip install arctic-inference[vllm]

This will install the latst Arctic Inference and compatible vLLM version.

Alternatively, you can also clone the Arctic Inference repository and
build/install it from source:

.. code-block:: bash

   git clone https://github.com/snowflakedb/ArcticInference.git && pip install ./ArcticInference

Serving
~~~~~~~

By using the examples below, you can get benefits from Shift Parallelism
Speculative Decoding, and SwiftKV all at once!

.. code-block:: bash

   vllm serve Snowflake/Llama-3.1-SwiftKV-8B-Instruct \
      --quantization "fp8" \
      --tensor-parallel-size 1 \
      --ulysses-sequence-parallel-size 2 \
      --enable-shift-parallel \
      --speculative-config '{
         "method": "arctic",
         "model":"Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
         "num_speculative_tokens": 3,
         "enable_suffix_decoding": true,
         "disable_by_batch_size": 64
      }'

Offline
~~~~~~~

.. code-block:: python

   import vllm
   from vllm import LLM, SamplingParams

   vllm.plugins.load_general_plugins()

   llm = LLM(
       model="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
       quantization="fp8",
       tensor_parallel_size=1,
       ulysses_sequence_parallel_size=2,
       enable_shift_parallel=True,
       speculative_config={
           "method": "arctic",
           "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
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

   sampling_params = SamplingParams(temperature=0.0, max_tokens=800)

   outputs = llm.chat(conversation, sampling_params=sampling_params)


.. toctree::
   :maxdepth: 1
   :caption: Arctic Inference
   :hidden:

   Home<self>

.. toctree::
   :maxdepth: 1
   :caption: Advanced Parallelism
   :hidden:

   shift-parallel
   arctic-ulysses

.. toctree::
   :maxdepth: 1
   :caption: Speculative Decoding
   :hidden:

   arctic-speculator
   suffix-decoding

.. toctree::
   :maxdepth: 1
   :caption: Model Optimization
   :hidden:

   swiftkv

.. toctree::
   :maxdepth: 1
   :caption: Optimized Embeddings
   :hidden:

   embeddings
