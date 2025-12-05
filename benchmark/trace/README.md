The patch allows running datasets in the following form:
```jsonl
{"timestamp": <ms>, "input_length": <tokens>, "output_length": <tokens>}
```
where each line corresponds to a request with random prompt to be sent at t = timestam.

The patch is applied as
```
bash apply_vllm_bench_patch_v10p1.sh
```
This will modify a couple of files of `vllm bench serve`. Please make sure vllm versions match.

Then use serve as
```
vllm bench serve --model $model --trace-dataset-path example_trace.jsonl --ignore-eos
```

