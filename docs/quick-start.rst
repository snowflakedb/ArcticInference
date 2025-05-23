
.. _quickstart:

===========
Quick Start
===========

To get started with ArcticInference optimization in vLLM, follow the steps below:

1. Install the ArcticInference package:

   .. code-block:: bash

      pip install arctic-inference[vllm]

2. Select the ArcticInference optimization(s) you want to use. You can
   choose one (or mix and match) the following optimizations:

   - :ref:`SwiftKV <swiftkv>`
   - :ref:`Arctic Ulysses <ulysses>`
   - :ref:`Speculative Decoding <spec-decode>`
   - :ref:`Shift Parallelism <shift>`
   - :ref:`Optimized Embeddings <embeddings>`

3. Add any necessary command-line arguments to your vLLM command. For example, to use
   Shift Parallelism, you would run:

   .. code-block:: bash

      python -m vllm.entrypoints.openai.api_server \
          ${vLLM_kwargs} \
          --shift-parallel-max-tp-size 8 \
          --shift-parallel-min-tp-size 1 \
          --shift-parallel-threshold 64
