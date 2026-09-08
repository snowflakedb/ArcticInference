from __future__ import annotations
import argparse
import inspect
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from types import ModuleType
import torch
from problem_loader import (Config, device_peak_bandwidth, device_peak_tflops, load_module,
                            local_rank, pick_device, rank, solution_class, world_size)
STAGES = ('compile', 'correctness', 'full')

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default
WARMUP_CALLS = 5           # untimed calls first: forces lazy alloc / autotune / JIT out
# Adaptive sampling bounds, used when a Config leaves `num_samples` as None. Sampling
# stops once the median's relative standard error falls under TARGET_RSE, so a quiet config
# stops at MIN and a variable one keeps going until it converges or hits MAX. A fixed count
# cannot do this: 12 was measured to stabilise batched_attention's largest config in
# isolation and still produced an 8.94% estimate in situ, because that config runs after
# four other multi-millisecond ones and the part is hotter by then.
# Floor and confirm-streak are measured, not guessed. Checking the criterion after every
# draw is repeated peeking: with ~50 chances the sample MAD eventually reads low and stops
# early. Simulated over 3000 trials per point, reporting the TRUE median-rse at stop as a
# multiple of the target (1.0 = exactly met):
#
#   floor confirm   2% spread          5% spread        n at 0.5%
#                   median  p95        median  p95
#       3       1     1.45  1.45         1.67  3.62            3
#       8       1     0.89  0.89         1.16  2.22            8
#       8       2     0.84  0.84         1.07  2.09            9   <- chosen
#      12       3     0.67  0.67         1.02  1.67           14
#
# floor=8 is where a 2%-spread config stops overshooting at all (a floor of 3 was wrong
# 88% of the time); confirm=2 buys most of what a longer streak would, for one extra draw.
# 12/3 is tighter still but costs 14 samples on the quiet configs, which is most of them.
# Residual weakness, stated rather than hidden: at genuinely high spread the p95 overshoot
# is ~2x, and SAMPLES_MAX is the only backstop.
SAMPLES_MIN = 8            # below this the MAD is too noisy to trust as a stopping signal
SAMPLES_CONFIRM = 2        # consecutive draws that must agree the median has converged
SAMPLES_MAX = 64           # ~5s on the most expensive config, counting input regeneration
TARGET_RSE = 0.01          # stop when the median is pinned to ~1%
CLOCK_WARMUP_S = 1.5       # run before timing: the GPU idles at ~120MHz vs 1515 loaded
CUPTI_BUFFER_BYTES = 8 * 1024 * 1024   # activity buffer CUPTI asks us to size
# Device activity counted as the solution's work. CONCURRENT_KERNEL alone is not enough:
# batched_attention's prefill holds one 11.97ms device MEMCPY, 15% of its 79.8ms. Under
# busy-time accounting a kernels-only measurement would not merely mis-bracket that op,
# it would drop it entirely -- the copy would read as idle and cost nothing.
CUPTI_KINDS = ('CONCURRENT_KERNEL', 'MEMCPY', 'MEMSET')

_L2_FLUSH: dict = {}       # device -> LLC-sized scratch, filled lazily by `_flush_l2`

class _TimedPathIncorrect(Exception):
    pass

class _NotMeasurable(Exception):
    """The solution ran but no device activity was attributed to it, so there is no
    latency to report. A hard error, never a correct-but-unscored result: a bogus
    latency would freeze the run's 1.0x baseline and be crowned as best."""
    pass

def _seed_rng(seed: int) -> None:
    """Seed torch per RANK, not per invocation.

    `torch.manual_seed` seeds every device identically, and the torchrun re-exec forwards
    the same `--seed` to every rank, so a shared seed makes all ranks draw the SAME
    inputs and weights. For a tensor-parallel problem that is not merely unrealistic --
    it makes the collective optional: with identical shards every rank's partial sum is
    identical, so `world_size * mm(x, w)` reproduces an all-reduce exactly and a solution
    can pass correctness while skipping the communication it is supposed to be measured
    on. Offsetting by rank keeps each invocation reproducible while giving every rank its
    own shard, which is also what real tensor parallelism looks like.
    """
    torch.manual_seed(seed + rank())

def _build(cls, config: Config, device: str):
    model = cls(*config.init_args, **config.init_kwargs)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    return model

def _robust_std(samples: Sequence[float], center: float) -> float:
    if len(samples) < 2:
        return 0.0
    mad = statistics.median([abs(s - center) for s in samples])
    return 1.4826 * mad

def _const_key(config: Config) -> str:
    """A config's compile-time identity: the arguments its solution is constructed with.

    `repr` rather than a tuple because `init_kwargs` values need not be hashable, and this
    only ever has to be equal for equal configurations -- never ordered or parsed.
    """
    return repr((config.init_args, sorted(config.init_kwargs.items())))

def _group_by_specialization(configs: Sequence[Config]) -> list[list[Config]]:
    """Configs sharing one constructor-argument set, in first-appearance order.

    One module is built per group and then handed every shape in it, so the constructor
    arguments are the only values a solution can treat as compile-time constants -- they
    are all `__init__` receives. Everything else arrives with the call.

    A problem that varies a value it passes through `init_kwargs` therefore puts every
    shape in a group of its own and gets no such pressure, which is that problem asserting
    the value really is fixed for a deployment.
    """
    groups: dict[str, list[Config]] = {}
    for config in configs:
        groups.setdefault(_const_key(config), []).append(config)
    return list(groups.values())

def _flush_l2(device: torch.device | str) -> None:
    """Evict the LLC, using twice its size so zeroing cannot leave any of it resident.

    In serving a kernel like this runs once per layer with a whole MoE layer's GEMMs in
    between, so it never finds its own data resident -- back-to-back calls are the
    unrealistically warm case. Sized from the device rather than hardcoded: the 256MiB
    this used to assume happened to cover current parts, which is luck, not a rule.

    One buffer per device per process; it is scratch with no relation to any solution.
    """
    buf = _L2_FLUSH.get(device)
    if buf is None:
        nbytes = torch.cuda.get_device_properties(device).L2_cache_size * 2
        buf = _L2_FLUSH[device] = torch.empty(nbytes, dtype=torch.int8, device=device)
    buf.zero_()

_CUPTI_RECS: list = []       # (start_ns, end_ns) per device op, filled by the callback
_CUPTI_KINDS: list = []      # resolved ActivityKind values for CUPTI_KINDS
_CUPTI_REGISTERED = False

def _cupti():
    """CUPTI module with our activity callbacks registered once per process.

    Registering per call was measured to *replace* rather than append (32 sessions
    yielded exactly 32 records), so it was harmless -- but it relies on undocumented
    behaviour, and appending would duplicate every record.
    """
    global _CUPTI_REGISTERED
    import cupti.cupti as cupti
    if not _CUPTI_REGISTERED:
        _CUPTI_KINDS.extend(getattr(cupti.ActivityKind, k) for k in CUPTI_KINDS)
        wanted = set(_CUPTI_KINDS)

        def requested():
            return CUPTI_BUFFER_BYTES, 0

        def completed(activities):
            for a in activities:
                if a.kind in wanted:
                    _CUPTI_RECS.append((a.start, a.end))

        cupti.activity_register_callbacks(requested, completed)
        _CUPTI_REGISTERED = True
    return cupti

def _merged_busy_ns(recs: Sequence[tuple[int, int]]) -> int:
    """Nanoseconds with at least one device op in flight: intervals sorted and merged.

    Merged rather than summed, so two overlapping kernels on separate streams are charged
    once. A raw sum reads two concurrent 10us kernels as 20us and penalizes a parallel
    solution 2x.
    """
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(recs):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total

# INVARIANT for everything below: every rank must execute the SAME sequence of
# collectives. A branch that is taken on one rank and not another -- a conditional reduce,
# an early return, a raise, or a loop whose trip count differs per rank -- does not produce
# a wrong answer, it HANGS until NCCL's watchdog aborts the communicator. Two such bugs
# (a time-bounded warmup loop and a conditional reduce) cost a 1005s hang before being
# found, so prefer reducing a flag and having all ranks act on the result.

def _distributed() -> bool:
    """Whether this process is one rank of a multi-GPU evaluation."""
    return world_size() > 1

def _free_port() -> int:
    """A port nobody is listening on, so concurrent evaluations cannot collide on the
    rendezvous. A fixed --master_port makes two overlapping runs hang."""
    import socket
    with socket.socket() as sock:
        sock.bind(('', 0))
        return int(sock.getsockname()[1])

def _relaunch_under_torchrun(num_gpus: int) -> None:
    """Re-exec this script under torchrun, once, when the problem wants several GPUs.

    Keeps the harness contract intact: the orchestrator still runs `python3 evaluate.py
    <problem> <solution> --stage ...` and still reads one JSON line from stdout. Nothing
    on the Rust side knows this happened, and `run_kernel.py` can do the same.

    `RANK` in the environment is the guard against re-execing forever -- torchrun sets it
    for every child, so a rank never takes this path.
    """
    torch.cuda.init()          # fail here, with a clear error, rather than inside a rank
    if num_gpus > torch.cuda.device_count():
        raise RuntimeError(
            f'problem asks for NUM_GPUS={num_gpus} but only {torch.cuda.device_count()} '
            'are visible'
        )
    os.execv(sys.executable, [
        sys.executable, '-m', 'torch.distributed.run',
        f'--nproc_per_node={num_gpus}',
        '--master_port', str(_free_port()),
        # Ranks other than 0 print nothing, so torchrun's per-rank log prefixes would only
        # ever decorate tracebacks. Keep stdout clean for the JSON.
        '--role', 'eval', '--tee', '0',
        os.path.abspath(__file__), *sys.argv[1:],
    ])

def _rendezvous(dev: torch.device) -> None:
    """Order every rank's next launch, without landing inside the measured window.

    A one-element device-side all-reduce, not `dist.barrier()`. Measured on 4xB200: a gloo
    CPU barrier made the same collective read 42.1us against 28.5us with this, and
    symmetric-memory all-reduce 51.2us against 26.2us -- a CPU barrier only orders the
    *launch*, so stream skew stays inside the window and gets attributed to the kernel.
    vLLM's newer benchmarks make the same switch for the same reason
    (benchmark_kimi_k3_gemm_rs_ar.py:432-435).
    """
    torch.distributed.all_reduce(torch.ones(1, device=dev, dtype=torch.float32))

def _agree_sample(ms: float | None, dev: torch.device) -> float | None:
    """One sample's latency on the critical path, agreed by every rank.

    Two reductions, and BOTH are unconditional -- see the invariant note at the top of the
    distributed helpers. MIN on a got-a-measurement flag, so a rank that captured no
    activity makes every rank discard the sample; MAX on the value, because a collective is
    only finished when its slowest participant is, and any single rank's clock understates
    it (rank 0's especially, which is what vLLM's older benchmarks report).

    Reducing also leaves every rank holding identical `latencies`, so the adaptive stopping
    rule reaches the same decision everywhere with no further communication.
    """
    got = torch.tensor([0.0 if ms is None else 1.0], dtype=torch.float64, device=dev)
    val = torch.tensor([0.0 if ms is None else ms], dtype=torch.float64, device=dev)
    torch.distributed.all_reduce(got, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.MAX)
    return float(val.item()) if got.item() > 0 else None

def _all_agree(ok: bool, reason: str | None) -> tuple[bool, str | None]:
    """Correctness is the AND over ranks; the reason comes from whichever rank objected."""
    flag = torch.tensor(1.0 if ok else 0.0, device='cuda', dtype=torch.float32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if flag.item() > 0:
        return (True, None)
    gathered: list = [None] * world_size()
    torch.distributed.all_gather_object(gathered, None if ok else f'rank {rank()}: {reason}')
    return (False, next((g for g in gathered if g), 'a rank reported incorrect'))

def _median_rse(latencies: Sequence[float]) -> float:
    """Relative standard error of the median of `latencies`.

    `_robust_std` is a MAD-derived sigma; the standard error of a *median* is
    `sqrt(pi/2) * sigma / sqrt(n)`, the 1.2533 factor being what a median gives up against
    a mean in exchange for not caring about outliers. Relative, so the threshold is
    dimensionless and one value serves a 60us kernel and a 20ms one alike.
    """
    n = len(latencies)
    if n < 2:
        return float('inf')
    med = statistics.median(latencies)
    if med <= 0:
        return float('inf')
    return 1.2533 * _robust_std(latencies, med) / math.sqrt(n) / med

def _busy_ms_window(t0: int, t1: int) -> float | None:
    """Merged device-busy ms among the activity records that fall inside [t0, t1].

    Records are collected for the whole sample loop and sliced per call afterwards, rather
    than enabling and disabling CUPTI around every call. Two reasons, and the second is the
    one that was actually costing us:

    * `activity_enable`/`flush_all`/`activity_disable` are host calls with variable cost. Run
      per sample they land AFTER the rank rendezvous, so each rank reaches its collective at
      a different moment; NCCL kernels spin while waiting, that spin is real device-busy
      time, and it gets charged to the kernel. TP4 read 73-99us per call this way against
      38us for the same work measured in a tight loop.
    * It keeps the LLC flush per call. Amortising N calls inside one window would be the
      easy fix for the overhead, but only the first of the N would see a cold cache, and
      several problems depend on that (glm gathers from a cache far larger than L2).

    `cupti.get_timestamp()` is the same clock the records carry, so the slice is exact.
    vLLM does this with bisect over sorted starts (timing.py:189-204).
    """
    inside = [(a, b) for a, b in _CUPTI_RECS if t0 <= a <= t1]
    if not inside:
        return None
    return _merged_busy_ns(inside) / 1e6

def _bench_solo(sol: Callable, config: Config, warmup_iters: int, num_samples: int | None) -> tuple[float, float]:
    """Median per-call latency in ms and its relative noise, for one config's shape.

    One freshly drawn input set per timed call, 1:1. No set is ever measured twice, so
    nothing a solution learns about one can carry into another, and the spread across them
    is a real spread over the inputs the kernel is expected to handle rather than a repeat
    of one draw.

    How many is adaptive unless the config pins `num_samples`: keep drawing until the
    median's relative standard error is under `TARGET_RSE`, bounded by SAMPLES_MIN/MAX. A
    quiet config stops at the floor and a variable one pays for as many as it needs, which
    is what a fixed count cannot do -- the right count depends on the config's own
    variability *and* on the thermal state it happens to run in.

    Sets are drawn inside the loop and freed, so memory stays at ~1 set no matter how many
    samples are asked for. That is only sound because the timing window wraps the `forward`
    call alone: `make_inputs` allocates *and* fills, which is real device work, but it
    completes and is flushed before the window opens.

    `sol` arrives already built and outlives this call: it is constructed once per
    constructor-argument set and handed every shape in that group, so whatever it does per
    shape is runtime adaptation rather than a compile-time constant. Warming here rather
    than at construction keeps a shape-triggered autotune out of the first sample.

    Only the candidate is timed; speedup against the baseline is derived in the
    orchestrator.
    """
    dist_on = _distributed()
    with torch.no_grad():
        warm = config.make_inputs()
        device = next((t.device for t in warm if torch.is_tensor(t)), 'cuda')
        for _ in range(WARMUP_CALLS):     # forces lazy alloc / autotune / JIT out of the
            sol(*warm)                    # measurement; a JIT would read as device time
        torch.cuda.synchronize()
        # Clocks next: idle is ~120MHz against ~1515 under load, so timing a cold part
        # measures a 12x ramp.
        if dist_on:
            # Count decided on rank 0 and broadcast. Time-bounding it per rank would have
            # each rank issue a different number of collectives -- a hang, not a skew.
            # Every rank runs the probe call, because it contains a collective too.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sol(*warm)
            torch.cuda.synchronize()
            per = max(time.perf_counter() - t0, 1e-6)
            n = torch.tensor([min(10_000, max(1, int(CLOCK_WARMUP_S / per)))],
                             dtype=torch.int64, device=device)
            torch.distributed.broadcast(n, src=0)
            for _ in range(int(n.item())):
                sol(*warm)
            torch.cuda.synchronize()
        else:
            deadline = time.perf_counter() + CLOCK_WARMUP_S
            while time.perf_counter() < deadline:
                for _ in range(max(1, warmup_iters)):
                    sol(*warm)
                torch.cuda.synchronize()
        del warm

        # `num_samples` pins the count; None means sample until the median converges.
        lo = SAMPLES_MIN if num_samples is None else max(2, num_samples)
        hi = SAMPLES_MAX if num_samples is None else lo
        latencies: list[float] = []
        streak = 0
        cupti = _cupti()
        _CUPTI_RECS.clear()
        for kind in _CUPTI_KINDS:            # enabled ONCE for the whole loop
            cupti.activity_enable(kind)
        try:
            while len(latencies) < hi:
                inputs = config.make_inputs()   # untimed: allocates and fills
                _flush_l2(device)               # after the fill, so it evicts that too
                torch.cuda.synchronize()        # ...and both land before the window opens
                if dist_on:
                    # Align the ranks, then sync so the rendezvous's own device work has
                    # finished and cannot land in this call's slice.
                    _rendezvous(device)
                    torch.cuda.synchronize()
                t0 = cupti.get_timestamp()
                sol(*inputs)
                torch.cuda.synchronize()
                t1 = cupti.get_timestamp()
                del inputs
                # Flushed per sample because the adaptive rule needs this call's value
                # before deciding whether to draw another. The cost lands AFTER t1 and
                # before the next iteration's rendezvous, so it is outside every slice and
                # no longer sits between the rendezvous and a measured call -- which is the
                # whole point of the change. `_CUPTI_RECS` keeps growing; the timestamp
                # slice ignores earlier samples' records.
                cupti.activity_flush_all(1)
                one_ms = _busy_ms_window(t0, t1)
                if dist_on:
                    one_ms = _agree_sample(one_ms, device)   # every rank, always
                if one_ms is not None:
                    latencies.append(one_ms)
                if len(latencies) >= lo and _median_rse(latencies) < TARGET_RSE:
                    streak += 1
                    if streak >= SAMPLES_CONFIRM:
                        break
                else:
                    streak = 0
        finally:
            for kind in _CUPTI_KINDS:
                cupti.activity_disable(kind)

    if not latencies:
        raise _NotMeasurable('no device activity was attributed to the solution')
    # Every sample is kept. The old loop discarded the first half, because 32 back-to-back
    # calls heat the part inside their own measurement window and clocks fall
    # 1515->1455MHz under sustained load. Regenerating the inputs between calls collapses
    # that duty cycle, and at the default of 3 samples a half-discard would leave one --
    # so the ramp is now handled by CLOCK_WARMUP_S alone, before timing starts.
    #
    # Median, not mean, even though each sample is a different input set. A stray context
    # switch or clock blip would drag a mean, and `rel_noise` is what discounts the score.
    # Mean would be the right summary of *expected* latency if the spread across sets were
    # the thing being reported; it is a one-line change if that is wanted.
    latency = statistics.median(latencies)
    return (latency, _robust_std(latencies, latency) / latency if latency > 0 else 0.0)

def _check_correctness(prob: ModuleType, solution_cls: type, device: str) -> tuple[bool, str | None]:
    checked = 0
    first_bad: str | None = None
    for group in _group_by_specialization(prob.make_configs()):
        shapes = [c for c in group if not getattr(c, 'perf_only', False)]
        if not shapes:
            continue
        # One instance per group, matching the timed path: a solution that is only correct
        # when constructed for a single shape fails here, at the stage that reports *why*,
        # rather than as a mystery in the benchmark. Constructor arguments are equal across
        # the group by construction, so `group[0]` builds it even if that entry is itself
        # perf_only.
        ref = _build(prob.Reference, group[0], device)
        sol = _build(solution_cls, group[0], device)
        if isinstance(ref, torch.nn.Module) and isinstance(sol, torch.nn.Module):
            sol.load_state_dict(ref.state_dict())
        for config in shapes:
            inputs = config.make_inputs()
            with torch.no_grad():
                ref_out, sol_out = (ref(*inputs), sol(*inputs))
            try:
                torch.testing.assert_close(sol_out, ref_out, **prob.TOLERANCE)
            except AssertionError as e:
                # Recorded, NOT returned. Under several ranks an early return leaves this
                # rank out of the next config's collectives, which hangs instead of
                # reporting the failure -- and a wrong solution is the common case.
                if first_bad is None:
                    first_bad = f'{config.name}: {str(e).splitlines()[0]}'
            checked += 1
    if checked == 0:
        # Raise rather than return correct=True: a problem where nothing is checked is an
        # authoring bug, and reporting "correct" for zero comparisons is the exact silent
        # success `perf_only` makes possible.
        raise ValueError('every config is perf_only, so nothing was correctness-checked')
    return (first_bad is None, first_bad)

def _assert_timed_path_correct(prob: ModuleType, ref: Callable, sol: Callable, config: Config) -> None:
    """Re-check the benchmarked instance after the timing loop. Not redundant with
    `_check_correctness`, which builds a fresh instance and so cannot see state the timing
    loop left behind -- accumulation into a buffer, or a cache built on the first call.
    Fresh inputs, not one of the samples it was timed on: a solution holding a timed
    sample's address is correct on that sample by construction, and wrong the moment it is
    handed another one.
    """
    probe = config.make_inputs()
    with torch.no_grad():
        ref_out, sol_out = (ref(*probe), sol(*probe))
    ok, why = True, None
    try:
        torch.testing.assert_close(sol_out, ref_out, **prob.TOLERANCE)
    except AssertionError as e:
        ok, why = False, f'{config.name}: timed-path output diverged: {str(e).splitlines()[0]}'
    if _distributed():
        # Agree before raising: one rank raising alone would abandon the others mid-run.
        ok, why = _all_agree(ok, why)
    if not ok:
        raise _TimedPathIncorrect(why or f'{config.name}: timed-path output diverged')

def _metric(fn, inputs, init_kwargs):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(inputs)
    if any((p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())):
        return fn(inputs, **init_kwargs)
    return fn(inputs, **{k: v for k, v in init_kwargs.items() if k in params})

def _add_roofline(entry, config, sol_ms, flops_fn, bytes_fn, peak_tflops, peak_gbps) -> None:
    if flops_fn is None and bytes_fn is None:
        return
    inputs = config.make_inputs()
    init_kwargs = getattr(config, 'init_kwargs', {}) or {}
    seconds = sol_ms / 1000.0
    if flops_fn is not None:
        tflops = float(_metric(flops_fn, inputs, init_kwargs)) / seconds / 1000000000000.0
        entry['achieved_tflops'] = tflops
        if peak_tflops:
            entry['pct_peak'] = tflops / peak_tflops
    if bytes_fn is not None:
        gbps = float(_metric(bytes_fn, inputs, init_kwargs)) / seconds / 1000000000.0
        entry['achieved_gbps'] = gbps
        if peak_gbps:
            entry['pct_bandwidth'] = gbps / peak_gbps

def _benchmark(prob: ModuleType, solution_cls: type, device: str, config_indices: Sequence[int] | None) -> dict:
    warmup_iters = getattr(prob, 'WARMUP_ITERS', 3)
    configs = prob.make_configs()
    if config_indices is not None:
        configs = [configs[i] for i in config_indices]
    peak_tflops = device_peak_tflops(device)
    peak_gbps = device_peak_bandwidth(device)
    flops_fn = getattr(prob, 'flops', None)
    bytes_fn = getattr(prob, 'bytes_moved', None)
    per_config = []
    for group in _group_by_specialization(configs):
        # One build per constructor-argument set, so those arguments are the only values
        # the solution can have baked in. Every shape in the group then goes to that one
        # instance: per-shape code has to be produced at runtime, which is what a served
        # kernel does anyway.
        #
        # `ref` before `sol`, keeping the RNG draw order `_check_correctness` uses, and
        # built once for the group -- its constructor arguments are what the group has in
        # common. It is only ever *run* on configs that are not perf_only, which is where
        # the oracle's memory is the binding limit; constructing it costs nothing shaped.
        needs_ref = any(not getattr(c, 'perf_only', False) for c in group)
        ref = _build(prob.Reference, group[0], device) if needs_ref else None
        sol = _build(solution_cls, group[0], device)
        if ref is not None and isinstance(ref, torch.nn.Module) and isinstance(sol, torch.nn.Module):
            sol.load_state_dict(ref.state_dict())
        for config in group:
            perf_only = getattr(config, 'perf_only', False)
            sol_ms, rel_noise = _bench_solo(sol, config, warmup_iters, config.num_samples)
            if not perf_only:
                _assert_timed_path_correct(prob, ref, sol, config)
            entry = {'name': config.name, 'latency_ms': sol_ms, 'latency_std_ms': rel_noise * sol_ms, 'rel_noise': rel_noise}
            if perf_only:
                # Informational only — Rust ignores unknown keys. Without it the journal
                # keeps no record of which scored latencies were never verified.
                entry['perf_only'] = True
            _add_roofline(entry, config, sol_ms, flops_fn, bytes_fn, peak_tflops, peak_gbps)
            per_config.append(entry)
    # Pure benchmark: absolute per-config latency + timing noise only. No speedup is
    # computed here — the reference is not timed, and speedup vs the baseline (the first
    # correct candidate) is derived in the orchestrator.
    metrics = {'per_config': per_config, 'noise_margin': max((c['rel_noise'] for c in per_config), default=0.0)}
    if flops_fn is not None and peak_tflops:
        metrics['peak_tflops'] = peak_tflops
    if bytes_fn is not None and peak_gbps:
        metrics['peak_gbps'] = peak_gbps
    return metrics

def evaluate(problem_path: str, solution_path: str, stage: str) -> dict:
    result = {'stage_requested': stage, 'stage_reached': 'load', 'ok': False, 'correct': False, 'error': None, 'device': None}
    try:
        device = pick_device()
        result['device'] = device
        prob = load_module(problem_path, 'problem')
        # A multi-GPU problem re-execs the whole script under torchrun and never returns
        # from here; the ranks it spawns re-enter with RANK set and fall through.
        num_gpus = int(getattr(prob, 'NUM_GPUS', 1))
        if num_gpus > 1 and 'RANK' not in os.environ:
            _relaunch_under_torchrun(num_gpus)
        if _distributed():
            torch.cuda.set_device(local_rank())      # before any problem code allocates
            torch.distributed.init_process_group(
                'nccl', device_id=torch.device('cuda', local_rank()))
        solution_cls = solution_class(load_module(solution_path, 'solution'))
        result['stage_reached'] = 'compile'
        _build(solution_cls, prob.make_configs()[0], device)
        if stage == 'compile':
            result['ok'] = True
            return result
        result['stage_reached'] = 'correctness'
        correct, reason = _check_correctness(prob, solution_cls, device)
        if _distributed():
            correct, reason = _all_agree(correct, reason)
        result['correct'] = correct
        if not correct:
            result['error'] = f'incorrect: {reason}'
            return result
        if stage == 'correctness':
            result['ok'] = True
            return result
        result['stage_reached'] = stage
        try:
            metrics = _benchmark(prob, solution_cls, device, None)
        except _TimedPathIncorrect as e:
            result['correct'] = False
            result['error'] = f'incorrect (timed path): {e}'
            return result
        result.update(metrics)
        result['ok'] = True
        return result
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback_tail'] = ''.join(traceback.format_exc().splitlines(keepends=True)[-6:])
        return result

def describe(problem_path: str) -> dict:
    """What a problem needs, for a scheduler that must size a job before running it.

    Read by the harness at startup:

        python3 evaluate.py --describe problems/tp_gemm_allreduce.py

    Python has to be the one answering. `NUM_GPUS` is a module attribute and a problem may
    compute it -- pattern-matching the source from Rust breaks the first time someone writes
    `NUM_GPUS = TP_SIZE`, and breaks silently, by reporting 1 for a problem that needs 4.

    `num_gpus` is answered without touching CUDA: importing the module is enough. Config
    names need `make_configs()`, which calls `pick_device()` and so requires a visible GPU
    -- reported when available and omitted when not, so a scheduler on a CPU-only host still
    learns the device count it has to plan for.
    """
    prob = load_module(problem_path, 'problem')
    out: dict = {'num_gpus': int(getattr(prob, 'NUM_GPUS', 1))}
    try:
        out['configs'] = [c.name for c in prob.make_configs()]
    except Exception as e:                       # no GPU, or an authoring error
        out['configs_unavailable'] = f'{type(e).__name__}: {e}'
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description='Staged AVO kernel evaluator')
    parser.add_argument('problem', help='path to problem.py')
    parser.add_argument('solution', nargs='?', help='path to the solution entrypoint, e.g. solution/solution.py')
    parser.add_argument('--describe', action='store_true',
                        help="print the problem's resource requirements as JSON and exit; "
                             'needs no solution, and answers num_gpus without a GPU')
    parser.add_argument('--stage', choices=STAGES, default='full')
    parser.add_argument('--seed', type=int, default=None, help="seed torch's RNG so this invocation's randomized inputs are reproducible (the orchestrator passes a fresh per-evaluation seed)")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.problem)))
        return
    if args.solution is None:
        parser.error('solution is required unless --describe is given')
    if args.seed is not None:
        _seed_rng(args.seed)
    out = evaluate(args.problem, args.solution, args.stage)
    # One JSON line on stdout whatever the world size: the orchestrator parses exactly one,
    # so non-zero ranks stay silent rather than interleaving four copies.
    if rank() == 0:
        print(json.dumps(out))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
if __name__ == '__main__':
    main()
