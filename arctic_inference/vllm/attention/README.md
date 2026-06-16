# Forest Cascade Attention (FCA)

Forest Cascade Attention discovers shared KV-cache prefix groups across
requests in each decode batch and splits each attention call into:

1. **Prefix FA** — one grouped call over the shared prefix blocks
   (`causal=False`), with all requests in a group treated as one sequence.
2. **Suffix FA** — a per-request causal call over the remaining (unique)
   KV blocks.
3. **Merge** — the two partial outputs are combined via log-sum-exp
   weighted merging (`merge_attn_states`).

This avoids redundant KV reads when many concurrent requests share long
prompt prefixes (e.g. system prompts, few-shot examples).

## Enabling FCA

Pass `--forest-cascade-attn-configs` with a JSON object to the vLLM
engine.  An empty `{}` enables FCA with all defaults:

```bash
vllm serve <model> \
    --forest-cascade-attn-configs '{}'
```

Custom tuning:

```bash
vllm serve <model> \
    --forest-cascade-attn-configs '{
        "max_query_len": 16,
        "min_group_size": 4,
        "min_additional_prefix_blocks": 2,
        "min_non_singleton_fraction": 0.3,
        "max_non_singleton_groups": 128
    }'
```

If `--forest-cascade-attn-configs` is **not** passed, forest cascade is
completely disabled and the backend behaves identically to upstream vLLM
FlashAttention.

## Configuration Reference

### `max_query_len`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `16` |

Maximum per-request query length (number of new tokens being decoded) for
forest cascade to activate.  When any request in the batch has a query
length exceeding this value, the entire batch falls back to standard
attention.

FCA is designed for decode-heavy batches where queries are short (typically
1 token per request).  Setting this higher allows FCA during chunked
prefill or multi-token speculative decoding, but very long queries reduce
the benefit of prefix sharing because the prefix FA call processes more
query tokens per group.

**Tuning:**
- For pure autoregressive decode: `1` is sufficient.
- With speculative decoding (e.g. `num_speculative_tokens=3`): set to
  `1 + num_speculative_tokens` (e.g. `4`).
- For mixed prefill+decode batches: `16` (the default) is a reasonable
  upper bound.

### `min_group_size`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `2` |

Minimum number of requests that must share a prefix for the group to be
considered "non-singleton" and eligible for the prefix FA call.  Groups
smaller than this threshold are treated as singletons — their prefix
blocks are folded into the suffix call instead.

This controls the trade-off between the overhead of an extra FA kernel
launch (prefix call) and the KV-read savings from grouping.

**Tuning:**
- `2` (default) is aggressive — even two requests sharing a prefix will
  trigger a grouped prefix call.  Good when prefixes are long.
- `4`–`8` reduces overhead from many tiny groups, useful when there are
  many distinct short prefixes.
- Higher values (e.g. `16`) make FCA very selective — it only fires when
  there are large clusters of shared-prefix requests.

### `min_additional_prefix_blocks`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |

Minimum number of *additional* shared KV blocks (beyond the global common
prefix) that a group must have for its prefix call to be worthwhile.  If a
group's shared prefix is not at least this many blocks longer than the
global common prefix, the group's prefix is collapsed to the common prefix
(effectively making it a singleton for the prefix call).

One KV block is typically 16 tokens (depending on `block_size`).

**Tuning:**
- `1` (default) means even one extra shared block (16 tokens) justifies
  a grouped prefix call.
- `2`–`4` requires 32–64 extra shared tokens, filtering out groups
  where the additional sharing is marginal.
- Increase if you see high CPU overhead from metadata building without
  measurable latency improvement (many groups with tiny prefixes).

### `min_non_singleton_fraction`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.25` |

Minimum fraction of requests in the batch that must belong to
non-singleton groups for FCA to activate.  If the grouping algorithm
finds shared prefixes for fewer than this fraction of requests, the
entire batch falls back to standard (or single-tree cascade) attention.

This is a global quality gate: if too few requests benefit from forest
cascade, the overhead of reordering, permuting Q, and running two FA
calls isn't worth it.

**Tuning:**
- `0.25` (default) means at least 25% of requests must be in
  non-trivial groups.
- Lower values (e.g. `0.1`) allow FCA to activate even when only a
  small fraction of the batch shares prefixes — useful if those
  prefixes are very long.
- Higher values (e.g. `0.5`) require a majority of the batch to benefit,
  which avoids FCA overhead in mixed workloads with mostly-unique prompts.

### `max_non_singleton_groups`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `256` |

Maximum number of non-singleton groups allowed.  If the grouping
algorithm produces more groups than this, FCA is disabled for the batch.
This bounds the CPU cost of metadata construction and the number of
"virtual sequences" in the prefix FA call.

**Tuning:**
- `256` (default) is generous and rarely hit.
- Reduce to `32`–`64` if CPU metadata overhead is noticeable at very
  high request counts (thousands of concurrent requests).
- In practice this knob rarely needs adjustment; the other parameters
  are more impactful.

## Implicit Eligibility Conditions

Beyond the configs above, FCA has built-in conditions that must all hold
for a given batch:

| Condition | Reason |
|---|---|
| `num_reqs >= 8` | Too few requests → grouping overhead exceeds benefit |
| `causal = True` | FCA only applies to causal (decoder) attention |
| `dcp_world_size <= 1` | Distributed context parallelism has its own KV partitioning |
| `cudagraph_mode` includes PIECEWISE (`PIECEWISE` or `FULL_AND_PIECEWISE`) | Forest cascade uses variable-shape metadata incompatible with full CUDA graph replay; with `FULL_AND_PIECEWISE` the cudagraph dispatcher picks PIECEWISE on batches where FCA fires and falls back to the captured FULL graph (regular FA) otherwise |
| `common_prefix_len` is block-aligned | The global common prefix must align to `block_size` boundaries |

When any condition is not met, the batch transparently falls back to
standard vLLM cascade attention (if a common prefix exists) or plain
FlashAttention.

## How It Works (Summary)

1. **Sort** requests by their KV block tables (lexicographic order on
   block IDs beyond the global common prefix).
2. **Group** by longest common prefix (LCP) using a recursive
   split-or-merge heuristic that maximizes `group_size * prefix_length`.
3. **Build metadata**: per-group prefix block tables, per-request suffix
   block tables, query permutations, and cumulative offsets.
4. **Prefix FA**: all queries are permuted into group order, then a
   single `flash_attn_varlen_func` call processes all groups as virtual
   sequences against their shared prefix blocks (`causal=False`).
5. **Suffix FA**: a second `flash_attn_varlen_func` call processes each
   request individually against its unique suffix blocks (`causal=True`).
6. **Merge**: `merge_attn_states` combines prefix and suffix outputs
   using log-sum-exp weighting.
7. **Scatter**: the merged output is written back to the original token
   order via `index_copy_`.
