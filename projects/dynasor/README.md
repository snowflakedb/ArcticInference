# Dynasor

Dynasor is a tool that helps you speed up LLM reasoning models without training or finetuning. It uses a combination of techniques to improve the prompt, and dynamically execute the prompt, and stop when the LLM has enough information to make a decision. 

For more details, see:
- [Blog post](https://hao-ai-lab.github.io/blogs/dynasor-cot/)
- [Paper](https://arxiv.org/abs/2412.20993)
- [Github (hao-ai-lab/Dynasor)](https://github.com/hao-ai-lab/Dynasor)

## Features

- Dynamic prompt execution with early stopping.
- Adaptive compute based on model certainty.
- General proxy server compatible with standard OpenAI API. 
- Support for batch and streaming responses


## Quick Start

### Option 1: One-click Server Setup

Start an arctic inference server (vLLM backend + OpenAI proxy server):
```bash
VLLM_USE_V1=1 python -m arctic_inference.dynasor.vllm_server \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  \
-tp 1 --enable-chunked-prefill --enforce-eager \
--port 8080
```

Start the vLLM client:
```bash
cd projects/dynasor/
python openai_client.py \
--base-url http://localhost:8080/v1 \
--probe-interval 32 \
--certainty-window 2 \
--prompt "2 + 2 ="
```

### Option 2: Separate Server Setup

Start a vLLM server:
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -tp 1 --enable-chunked-prefill --enforce-eager
```

Start the proxy server:
```bash
python -m arctic_inference.dynasor.openai_server \
--target-base-url http://localhost:8000 \
--port 8080
```

Start the vLLM client:
```bash
cd projects/dynasor/
python openai_client.py \
--base-url http://localhost:8080/v1 \
--probe-interval 32 \
--certainty-window 2 \
--prompt "2 + 2 ="
```

## API Usage

The Dynasor client implements the OpenAI API format, making it easy to integrate with existing applications. Here's a basic example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your prompt here"}
    ],
    model="your-model",
    max_tokens=2048,
    extra_body={
        # Optional: Dynasor parameters
        "dynasor": {
            "probe_interval": 32,
            "certainty_window": 2
        }
    },
    stream=True
)
```

## References

- [Blog post](https://hao-ai-lab.github.io/blogs/dynasor-cot/)
- [Paper](https://arxiv.org/abs/2412.20993)
- [Github (hao-ai-lab/Dynasor)](https://github.com/hao-ai-lab/Dynasor)

```bibtex
@article{fu2024efficiently,
  title={Efficiently Serving LLM Reasoning Programs with Certaindex},
  author={Fu, Yichao and Chen, Junda and Zhu, Siqi and Fu, Zheyu and Dai, Zhongdongming and Qiao, Aurick and Zhang, Hao},
  journal={arXiv preprint arXiv:2412.20993},
  year={2024}
}
```