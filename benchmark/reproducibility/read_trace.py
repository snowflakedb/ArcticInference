import json
import csv
from datetime import datetime
from collections import defaultdict

def parse_timestamp(timestamp_str):
    """Parse timestamp string and return datetime object."""
    try:
        # Truncate microseconds to 6 digits if longer
        if '.' in timestamp_str:
            parts = timestamp_str.split('.')
            if len(parts[1]) > 6:
                timestamp_str = parts[0] + '.' + parts[1][:6]
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # Try without microseconds if the format doesn't match
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def convert_to_relative_ms(timestamps):
    """Convert list of timestamp strings to milliseconds relative to first entry."""
    if not timestamps:
        return []
    
    first_time = parse_timestamp(timestamps[0])
    relative_ms = []
    
    for ts in timestamps:
        current_time = parse_timestamp(ts)
        delta = (current_time - first_time).total_seconds() * 1000
        relative_ms.append(delta)
    
    return relative_ms

def read_csv(file_path):
    data = []
    timestamps = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row.get("TIMESTAMP"))
            entry = {
                "timestamp": row.get("TIMESTAMP"),
                "input_length": int(row.get("ContextTokens")),
                "output_length": int(row.get("GeneratedTokens"))
            }
            data.append(entry)
    
    # Convert timestamps to relative milliseconds
    relative_ms = convert_to_relative_ms(timestamps)
    for i, entry in enumerate(data):
        entry["timestamp"] = relative_ms[i]
    
    return data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # Extract required columns
            entry = {
                "timestamp": obj.get("timestamp"),
                "input_length": obj.get("input_length"),
                "output_length": obj.get("output_length")
            }
            data.append(entry)
    return data

# Example usage:
# file_path = "synthetic_trace.jsonl"
# file_path = "toolagent_trace.jsonl"
file_path = "conversation_trace.jsonl"
records = read_jsonl(file_path)

# file_path = "AzureLLMInferenceTrace_conv.csv"
# file_path = "AzureLLMInferenceTrace_code.csv"
# records = read_csv(file_path)

# file_path = "AzureLLMInferenceTrace_code_1min_section.jsonl"
# records = read_jsonl(file_path)


# threshold_low =  2 * 60 * 1000
# threshold_high = 17 * 60 * 1000
threshold_low =  0 * 60 * 1000
threshold_high = 15 * 60 * 1000
records = [record for record in records if threshold_low <= record["timestamp"] <= threshold_high]

# print(f"Filtered records between {threshold_low} ms and {threshold_high} ms: {len(records)} entries.")

num_prompts = len(records)
timestamp = [int(record["timestamp"]) for record in records]
timestamp = [ts - timestamp[0] for ts in timestamp]  # Normalize to start from 0 ms
input_length = [record["input_length"] for record in records]
output_length = [record["output_length"] for record in records]


# Calculate sum of input_length per batch (prompts with same timestamp)

batch_sums = defaultdict(int)
batch_counts = defaultdict(int)

for req in range(num_prompts):
    ts = timestamp[req]
    batch_sums[ts] += input_length[req]
    batch_counts[ts] += 1

# Print batch statistics
print(f"\nBatch statistics (prompts with same timestamp):")
print(f"Number of unique timestamps (batches): {len(batch_sums)}")
for ts in sorted(batch_sums.keys()):
    print(f"Timestamp {ts} ms: {batch_counts[ts]} prompts, total input_length = {batch_sums[ts]}")
print(f"max input_length in a batch: {max(batch_sums.values())}\n")

# # Write to jsonl file
# output_file = file_path.replace('.jsonl', '_processed_15mins.jsonl').replace('.csv', '_processed_15mins.jsonl')
# with open(output_file, 'w') as f:
#     for req in range(num_prompts):
#         entry = {
#             "timestamp": timestamp[req],
#             "input_length": input_length[req],
#             "output_length": output_length[req]
#         }
#         f.write(json.dumps(entry) + '\n')
print(f"timestamp input_length output_length")
for req in range(num_prompts):
    # print(f"Record {req}: Timestamp={timestamp[req]}, Input Length={input_length[req]}, Output Length={output_length[req]}")
    print(f"{timestamp[req]} {input_length[req]} {output_length[req]}")

import matplotlib.pyplot as plt

window_size = 80000 # in ms
request_rate = []
time_axis = [start_time/1000 for start_time in range(0, max(timestamp)+window_size, window_size)]
for start_time in range(0, max(timestamp)+window_size, window_size):
    new_requests = [sample for sample in timestamp if start_time <= sample < start_time+window_size]
    request_rate.append(len(new_requests)/(window_size/1000))

plt.figure(figsize=(10, 6))
plt.plot(time_axis, request_rate, linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Request Rate (requests/second)')
plt.title(f'Request Rate over Time {file_path}')
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(range(num_prompts), input_length, linestyle='-')
# plt.xlabel('Request Index')
# plt.ylabel('Input Length')
# plt.title(f'Input Length over Requests {file_path}')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(num_prompts), output_length, linestyle='-')
plt.xlabel('Request Index')
plt.ylabel('Output Length')
plt.title(f'Output Length over Requests {file_path}')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(num_prompts), timestamp, linestyle='-')
plt.xlabel('Request Index')
plt.ylabel('Timestamp')
plt.title(f'Request Timestamps {file_path}')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(output_length, input_length, alpha=0.5)
plt.xlabel('Output Length')
plt.ylabel('Input Length')
plt.yscale('log')
plt.title(f'Input Length vs Output Length {file_path}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Total number of prompts: {num_prompts} in {file_path}")