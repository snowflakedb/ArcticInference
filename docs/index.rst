
ArcticInference documentation
=============================

ArcticInference is a new library from Snowflake AI Research that contains
current and future LLM inference optimizations developed at Snowflake. It is
integrated with vLLM v0.8.4 using vLLM's custom plugin feature, allowing us to
develop and integrate inference optimizations quickly into vLLM and make them
available to the community.

Once installed, ArcticInference automatically patches vLLM to use Arctic Ulysses
and other optimizations implemented in ArcticInference, and users can continue
to use their familiar vLLM APIs and CLI. It's easy to get started!

Key Features
------------

🚀 :ref:`SwiftKV <swiftkv>`: Reduce compute during prefill by reusing key-value pairs across transformer layers

🚀 :ref:`Arctic Ulysses <ulysses>`: Improve long-context inference latency and throughput via sequence parallelism across GPUs

🚀 :ref:`Speculative Decoding <spec-decode>`: Boosts LLM speed by drafting tokens with a small model and verifying them in bulk

🚀 :ref:`Shift Parallelism <shift>`: <add description of Shift Parallelism here>

🚀 :ref:`Optimized Embeddings <embeddings>`: <add description of Optimized Embeddings here>

Quick Start
-----------

To get started with ArcticInference check out the :ref:`quick start guide <quickstart>`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   install
   swiftkv
   ulysses
   spec-decode
   shift
   embeddings
