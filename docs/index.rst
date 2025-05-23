
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

- **SwiftKV**: <add description of SwiftKV here>
- **Arctic Ulysses**: <add description of Arctic Ulysses here>
- **Speculative Decoding**: <add description of Speculative Decoding here>
- **Shift Parallelism**: <add description of Shift Parallelism here>
- **Optimized Embeddings**: <add description of Optimized Embeddings here>

Quick Start
-----------

To get started with ArcticInference check out the :ref:`quick start guide <quickstart>`

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick-start
   install
   swiftkv
   ulysses
   spec-decode
   shift
   embeddings
