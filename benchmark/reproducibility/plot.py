import pandas as pd
import matplotlib.pyplot as plt

# modify the path to your CSV file as needed
df = pd.read_csv("code_output.csv")

request = df["request"]
timestamp = df["timestamp"]
input_length = df["input_len"]
output_length = df["output_len"]
ttft = df["ttft_ms"]
tpot = df["tpot_ms"]
latency = df["e2el_ms"]

# TTFT
plt.figure()
plt.plot(ttft)
plt.xlabel("Request Id")
plt.ylabel("Milliseconds")
plt.title("TTFT")
plt.savefig("ttft.png")

# TPOT
plt.figure()
plt.plot(tpot, 'o')
plt.yscale('log')
plt.xlabel("Request Id")
plt.ylabel("Milliseconds")
plt.title("TPOT")
plt.savefig("tpot.png")

# Latency
plt.figure()
plt.plot(latency, 'o')
plt.xlabel("Request Id")
plt.ylabel("Milliseconds")
plt.title("Completion Time")
plt.savefig("latency.png")

