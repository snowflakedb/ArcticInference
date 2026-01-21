
## Reproducibility

This page is on reproducibility of the [Shift Parallelism](https://arxiv.org/pdf/2509.16495) paper. Please see the Artifact Appendix.

## Instructions

### Step 1: Making vLLM work

1. Create a clean environment

- If python 3.10 is already installed, create a clean virtual environment
```console
python3.10 -m venv myvenv
source myvenv/bin/activate
```

- If not, use conda to create an environment with python 3.10
```console
conda create -n myenv python=3.10
conda activate myenv
```

2. Install vLLM
```console
pip install vllm==v0.10.1
```

3. Download the models
```console
huggingface-cli download RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --local-dir RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
huggingface-cli download Qwen/Qwen3-32B-FP8 --local-dir Qwen/Qwen3-32B-FP8
```

### Step 2: Install ArcticInference

```console
cd ArcticInference
pip install .
```

Once installed, Arctic Inference automatically patches vLLM to use Arctic Inference with Shift Parallelism, and users can continue to use their familiar vLLM APIs and CLI.

### Step 3: Vibe test
`vibe_test.py`
```python
import vllm
from vllm import LLM, SamplingParams

vllm.plugins.load_general_plugins()

llm = LLM(
    model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
    ulysses_sequence_parallel_size=8,
    enable_shift_parallel=True,
    shift_parallel_threshold=8,
)

conversation = [
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=800)

outputs = llm.chat(conversation, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

```console
ARCTIC_INFERENCE_ENABLED=1 VLLM_DISABLE_COMPILE_CACHE=1 python vibe_test.py
```
We turn off compile cache for now to prevent complications.

### Step 4: Patch vLLM bench

This patch is necessary to run traces for running traces.

Please replace your `vllm/bencmarks/serve.py` with `patches/serve.py` and `vllm/benchmarks/datasets.py` with `patches/datasets.py`. The vLLM path can be found by `pip show vllm`.

### Step 5: Run Traces

The traces are available: https://doi.org/10.5281/zenodo.18240909

```console
wget https://zenodo.org/records/18240909/files/AzureLLMInferenceTrace_code_15mins.jsonl
wget https://zenodo.org/records/18240909/files/conversation_trace_15mins.jsonl
```

There are two traces and three parallelisms in full reproduction. For each case, the server and the client is run in separate terminals. For example,

1. Azure LLM code (Figure 9)

server:
```console
ARCTIC_INFERENCE_ENABLED=1 VLLM_DISABLE_COMPILE_CACHE=1 vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --ulysses-sequence-parallel-size 8 \
  --enable-shift-parallel \
  --max-num-batched-tokens 131072
```

client:
```console
vllm bench serve --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --trace-dataset-path AzureLLMInferenceTrace_code_15mins.jsonl --ignore-eos --trace-output-path code_output.csv
```

2. Mooncake conversation (Figure 10)

For the Mooncake dataset, we first increase the context length with YaRN: https://huggingface.co/Qwen/Qwen3-32B-FP8#processing-long-texts

server:
```console
ARCTIC_INFERENCE_ENABLED=1 VLLM_DISABLE_COMPILE_CACHE=1 vllm serve Qwen/Qwen3-32B-FP8 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --ulysses-sequence-parallel-size 8 \
  --enable-shift-parallel \
  --max-num-batched-tokens 131072
```

client:
```console
vllm bench serve --model Qwen/Qwen3-32B-FP8 --trace-dataset-path conversation_trace_15mins.jsonl --ignore-eos --trace-output-path conversation_output.csv
```

Ready-to-use `server.sh` and `client.sh` are included.

The key results involves two figures (Figure 9—10). The breakdown of reproduction times are given below:

Figure 9: DP (15 mins), TP (15 mins), SP (15 mins), Shift Parallel (15 mins)
Figure 10: DP (~2.5 hrs), TP (~1 hr), SP (15 mins), Shift Parallel (15 mins)

### Step 6: Plotting

Install plotting library.
```console
pip install matplotlib
```

run `plot.py` with appropriate trace output path.
