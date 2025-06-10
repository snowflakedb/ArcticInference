
Arctic Inference documentation
==============================

Arctic Inference is a new library from Snowflake AI Research that contains
current and future LLM inference optimizations developed at Snowflake. It is
integrated with vLLM v0.8.4 using vLLM's custom plugin feature, allowing us to
develop and integrate inference optimizations quickly into vLLM and make them
available to the community.

Once installed, Arctic Inference automatically patches vLLM to use Arctic Ulysses
and other optimizations implemented in Arctic Inference, and users can continue
to use their familiar vLLM APIs and CLI. It's easy to get started!

Key Features
------------

Advanced Parallelism
~~~~~~~~~~~~~~~~~~~~

🚀 :ref:`shift`:
   Dynamically switches between tensor and sequence parallelism at runtime to optimize latency, throughput, and cost — all in one deployment

🚀 :ref:`ulysses`:
   Improve long-context inference latency and throughput via sequence parallelism across GPUs

Speculative Decoding
~~~~~~~~~~~~~~~~~~~~

🚀 :ref:`arctic-speculator`:
   Lightweight yet effective draft models based on MLP and LSTM architectures, complete with training pipelines

🚀 :ref:`suffix-decoding`:
   Rapid speculation for long repeated sequences, effective for coding, agents, and other emerging applications

Model Optimization
~~~~~~~~~~~~~~~~~~

🚀 :ref:`swiftkv`:
   Reduce compute during prefill by reusing key-value pairs across transformer layers

Other Optimizations
~~~~~~~~~~~~~~~~~~~

🚀 :ref:`embeddings`:
   Accelerate embedding performance with parallel tokenization, byte outputs, and GPU load-balanced replicas

Quick Start
-----------

To get started with Arctic Inference check out the :ref:`quick start guide <quickstart>`

Table of Contents
=================

.. toctree::
   :maxdepth: 1

   quick-start
   install

.. toctree::
   :maxdepth: 1
   :caption: Advanced Parallelism

   shift
   ulysses

.. toctree::
   :maxdepth: 1
   :caption: Speculative Decoding

   arctic-speculator
   suffix-decoding

.. toctree::
   :maxdepth: 1
   :caption: Model Optimization

   swiftkv

.. toctree::
   :maxdepth: 1
   :caption: Optimized Embeddings

   embeddings
