
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

Optimized Generative AI

🚀 :ref:`Shift Parallelism <shift>`:
   Dynamically switches between tensor and sequence parallelism at runtime to optimize latency, throughput, and cost — all in one deployment

🚀 :ref:`Arctic Ulysses (Sequence Parallelism) <ulysses>`:
   Improve long-context inference latency and throughput via sequence parallelism across GPUs

🚀 :ref:`Speculative Decoding <spec-decode>`:
   Boosts LLM speed by drafting tokens with a small model and verifying them in bulk

🚀 :ref:`SwiftKV <swiftkv>`:
   Reduce compute during prefill by reusing key-value pairs across transformer layers

Optimized Embeddings

🚀 :ref:`Optimized Embeddings <embeddings>`:
   <add description of Optimized Embeddings here>

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
   :caption: Optimized Generative AI

   shift
   ulysses
   spec-decode
   swiftkv

.. toctree::
   :maxdepth: 1
   :caption: Optimized Embeddings

   embeddings
