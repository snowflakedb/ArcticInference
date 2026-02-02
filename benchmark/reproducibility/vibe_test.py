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

