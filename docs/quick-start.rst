
.. _quickstart:

===========
Quick Start
===========

To get started with Arctic Inference optimization in vLLM, follow the steps below:

1. Install the Arctic Inference package:

   .. code-block:: bash

      pip install arctic-inference[vllm]

2. Select the Arctic Inference optimization(s) you want to use. You can
   choose one (or mix and match) the following optimizations:

   - Optimized Generative AI:

     - :ref:`shift`
     - :ref:`ulysses`
     - :ref:`spec-decode`
     - :ref:`swiftkv`

   - Optimized Embeddings:

     - :ref:`embeddings`

3. Add any necessary command-line arguments to your vLLM command. For example, to use
   Shift Parallelism, you would run:

   .. code-block:: bash

      python -m vllm.entrypoints.openai.api_server \
          ${vLLM_kwargs} \
          --ulysses-sequence-parallel-size 8 \
          --enable-shift-parallel
