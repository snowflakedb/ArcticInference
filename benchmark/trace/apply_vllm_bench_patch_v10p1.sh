#!/usr/bin/env bash
set -euo pipefail

SERVE_PY="/home/yak/myvenv/lib/python3.10/site-packages/vllm/benchmarks/serve.py"
DATASETS_PY="/home/yak/myvenv/lib/python3.10/site-packages/vllm/benchmarks/datasets.py"

patch "$DATASETS_PY" << 'EOF'
79d78
<     trace_timestamp: int = 0
338d336
<         trace_dataset_path = None,
353,373c351,367
<         print(f"RandomDataset sample {trace_dataset_path}")
< 
<         events = []
<         with open(trace_dataset_path, "r") as f:
<             for line in f:
<                 if not line.strip():
<                     continue
<                 obj = json.loads(line)
<                 timestamp = obj["timestamp"]
<                 input_length = obj["input_length"]
<                 output_length = obj["output_length"]
<                 print(f"read trace timestamp {timestamp} input_length {input_length} output_length {output_length}")
<                 events.append((timestamp, input_length, output_length))
<         # Ensure chronological order
<         events.sort(key=lambda x: x[0])
<         print(f"events {events}")
< 
<         num_requests = len(events)
<         timestamps = [i[0] for i in events]
<         input_lens = [i[1] for i in events]
<         output_lens = [i[2] for i in events]
---
>         # New sampling logic: [X * (1 - b), X * (1 + b)]
>         input_low = int(real_input_len * (1 - range_ratio))
>         input_high = int(real_input_len * (1 + range_ratio))
>         output_low = int(output_len * (1 - range_ratio))
>         output_high = int(output_len * (1 + range_ratio))
> 
>         # Add logging for debugging
>         logger.info(
>             "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
>             input_low, input_high, output_low, output_high)
> 
>         input_lens = np.random.randint(input_low,
>                                        input_high + 1,
>                                        size=num_requests)
>         output_lens = np.random.randint(output_low,
>                                         output_high + 1,
>                                         size=num_requests)
400d393
<                     trace_timestamp=timestamps[i]
765d757
<                 trace_dataset_path=args.trace_dataset_path,
EOF

patch "$SERVE_PY" << 'EOF'
101,116d100
< async def get_request_trace(input_requests: list[SampleRequest]) -> AsyncGenerator[tuple[SampleRequest, float], None]:
< 
<     print(f"get_request_trace")
< 
<     prev_ts = 0
<     for i in range(len(input_requests)):
<         request = input_requests[i]
<         curr_ts = request.trace_timestamp
<         delay = curr_ts - prev_ts
<         print(f"request {i} prompt_len {request.prompt_len} expected_output_len {request.expected_output_len} timestamp {request.trace_timestamp} delay {delay} ms")
<         if delay > 0:
<             await asyncio.sleep(delay/1000)
<         prev_ts = curr_ts
<         yield request, 1
< 
< 
267,270d250
<     print(f"requsest, timestamp, input_len, output_len, ttft_ms, tpot_ms, e2el_ms")
<     for i in range(len(outputs)):
<         print(f"{i} {input_requests[i].trace_timestamp} {input_requests[i].prompt_len} {actual_output_lens[i]} {ttfts[i]*1000:.2f} {all_tpots[i]*1000:.2f} {e2els[i]*1000:.2f}")
< 
356d335
<     trace_dataset_path: str | None = None,
477,488d455
<     if trace_dataset_path is not None:
<         iterator = get_request_trace(input_requests)
<     else:
<         iterator = get_request(
<             input_requests,
<             request_rate,
<             burstiness,
<             ramp_up_strategy,
<             ramp_up_start_rps,
<             ramp_up_end_rps,
<         )
< 
501c468,470
<     async for request, current_request_rate in iterator:
---
>     async for request, current_request_rate in get_request(
>             input_requests, request_rate, burstiness, ramp_up_strategy,
>             ramp_up_start_rps, ramp_up_end_rps):
987d955
<     parser.add_argument("--trace-dataset-path", type=str, default=None)
1096d1063
<             trace_dataset_path=args.trace_dataset_path,
EOF

echo "Patch applied successfully!"

echo "You can now use:"
echo ""
echo "  vllm bench serve \\"
echo "      --model RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 \\"
echo "      --trace-dataset-path example_trace.jsonl"
echo ""
