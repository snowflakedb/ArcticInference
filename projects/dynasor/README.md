# Dyansor

## Quick Start (Chat-based Interface)

```bash
cd arctic_inference/projects/dynasor/
```

Start the server:
```bash
python api_serve.py \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--port 8000 \
-tp 1 --enable-prefix-caching --enable-chunked-prefill
```

Start the chat client, the UI we designed to interact with the server:
```bash
python chat.py \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--base-url http://localhost:8000/v1
```

Then in the chat interface, ask a simple question:
```
> 2+2=?
```

## Quick Start (vLLM Client)

```bash
cd arctic_inference/projects/dynasor/
```

Start the server:
```bash
python api_serve.py \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--port 8000 \
-tp 1 --enable-prefix-caching --enable-chunked-prefill
```

Start the vLLM client:
```bash
python examples/vllm_client.py
```
