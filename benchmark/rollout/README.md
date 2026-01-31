
This patch extencds SamplingParams to specify the length of each sequence when n > 1.

The patch is applied as `source patch_sampling.sh`.

As a result, you can specify `max_tokens_n` as a list in sampling params and set `ignore_eos` so that each sequence generates exactly specified number of tokens.

```
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]

sampling_params = [SamplingParams(n=2,
                                 temperature=0.8,
                                 top_p=1.0,
                                 max_tokens_n=[25, 50],
                                 ignore_eos=True,
                                ),
                   SamplingParams(n=3,
                                 temperature=0.8,
                                 top_p=1.0,
                                 max_tokens_n=[5, 10, 15],
                                 ignore_eos=True,
                                ),
                   SamplingParams(n=1,
                                 temperature=0.8,
                                 top_p=1.0,
                                 max_tokens=100,
                                 ignore_eos=True,
                                 # max_tokens_n=[100], this will be ineffective since n = 1
                                ),
                   ]

outputs = llm.generate(prompts, sampling_params=sampling_params)
```

The number of resulting input and output tokens per sequence:
```
prompt 0 seq 0: input 5 output 25
prompt 0 seq 1: input 5 output 50
prompt 1 seq 0: input 7 output 5
prompt 1 seq 1: input 7 output 10
prompt 1 seq 2: input 7 output 15
prompt 2 seq 0: input 5 output 100
```

