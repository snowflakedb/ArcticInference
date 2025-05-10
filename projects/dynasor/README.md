## Dyansor

Dynasor is a tool that helps you speed up LLM reasoning model without training or finetuning. It uses a combination of techniques to improve the prompt, and dynamically execute the prompt, and stop when the LLM has enough information to make a decision. 

For more details, see:
- [Blog post](https://hao-ai-lab.github.io/blogs/dynasor-cot/)
- [Paper](https://arxiv.org/abs/2412.20993)
- [Github (hao-ai-lab/Dynasor)](https://github.com/hao-ai-lab/Dynasor)


### Quick Start

Start the server:
```bash
python -m arctic_inference.dynasor.api_serve \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--port 8000 \
-tp 1 --enable-prefix-caching --enable-chunked-prefill --enforce-eager
```

Start the vLLM client:
```bash
cd arctic_inference/projects/dynasor/
python vllm_client.py
```
