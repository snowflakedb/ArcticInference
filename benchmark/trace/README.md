The patch allows running datasets in the following form:
```jsonl
{"timestamp": 15, "input_length": 1000, "output_length": 128}
```
where each line corresponds to a request with random prompt to be sent at 15 milliseconds from t = 0.

The patch is applied as
```
bash apply_vllm_bench_patch_v10p1.sh
```
This will modify a couple of files of `vllm bench serve`. Please make sure vllm versions match.

Then use serve as
```
vllm bench serve --model $model --trace-dataset-path example_trace.jsonl --ignore-eos
```

