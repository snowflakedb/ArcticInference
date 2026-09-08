# reproduce

`example_full.py` walks each model through a cold start, then the `saved -> up`
trip (sleep, CUDA checkpoint, CRIU dump/restore, NCCL reinit, weight restore,
graph recapture). It covers TP1 / TP2 / TP4 / TP8.

Must run as **root** (CRIU and `cuda-checkpoint`). From this package directory:

```bash
python ./reproduce/example_full.py                 # all models
python ./reproduce/example_full.py Qwen3.5-2B-tp1  # one label
```

Weights are read from `/data-fast/hf_models/<org>/<name>` when that path exists,
otherwise from Hugging Face. Images go under
`/data-fast/image-cache/reproduce_example_full/<label>`.

Each model prints per-primitive seconds after `saved -> up`. The same numbers
are printed again under `== times ==` at the end of the run.
