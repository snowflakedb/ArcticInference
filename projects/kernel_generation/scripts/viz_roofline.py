#!/usr/bin/env python3
"""Roofline plots for a kernelguy run.

Usage: viz_roofline.py <run_dir> [out_dir]   (out_dir defaults to <run_dir>)

Reads <run_dir>/history/nodes.jsonl READ-ONLY and emits into out_dir:
  - viz_roofline.png        every config placed on the device roofline (log-log)
  - viz_compute_tflops.png  achieved TFLOP/s over wall time, compute-bound configs
  - viz_memory_bw.png       achieved GB/s over wall time, memory-bound configs

Which ceiling binds a config is physics, not naming. Arithmetic intensity falls out of
the two numbers the evaluator already records per config (`scripts/evaluate.py`'s
`_add_roofline`):

    FLOP/byte = 1000 * achieved_tflops / achieved_gbps

The elapsed time cancels, so intensity is a property of the config rather than of the
measurement. Above the device ridge point a config is compute-bound and belongs on the
TFLOP/s axis; below it, bandwidth is the limit and its TFLOP/s figure says nothing
about the kernel. So this works for any problem that defines `flops` and `bytes_moved`,
whatever it calls its configs.

Both plots agree on `% of attainable` by construction: for a memory-bound config,
achieved_tflops / (intensity * peak_gbps) reduces exactly to achieved_gbps / peak_gbps.

The time-series curves track the CONFIRMED best solution so far, not each config's own
maximum: they step only when a solution is promoted, and then every curve takes that
one solution's numbers. So a vertical slice is always a single real solution that
survived re-measurement -- which means a curve can step down, when the new overall
winner happens to be slower on that one config. Every individual timed attempt stays
behind as a scatter, so the spread and the attempt count are still visible, including
the optimistic ones the search rejected.

With <run_dir>/reference.json present (an evaluator JSON from running the problem's own
reference as a solution) each config also gets its reference as a dotted line in its own
colour, and the legend carries the ratio -- which is the question actually being asked:
whether the kernel beat the obvious implementation, not how large the run's self-relative
geomean grew.

Device peaks are derived from achieved/pct, which the evaluator records whenever the
device is in `scripts/problem_loader.py`'s tables. Without them there is no ridge and so
no roofline: the two time-series plots still draw, ceiling-less, over every config, and
say why.

Safe to run against a live run dir: it only reads, and a partially-written trailing
JSONL line is skipped.
"""
import json, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe

PEAK_RED = "#d03b3b"                 # the ceiling is a hard limit, so it reads as one
LEGEND_INK = "#52514e"               # the style key is a note, not a series: no hue
SURFACE = "#ffffff"                  # marker rings, so overlapping points stay countable
# Eight categorical hues in fixed order, shared with viz_run.py. Colour is config
# IDENTITY, so it is assigned once over every config name in the run and never per
# plot: a config drawn on two plots has to be the same colour on both. Marker shape is
# the second channel -- adjacent-pair CVD separation on this palette is comfortable,
# but three of the hues sit under 3:1 against white, which the direct labels on the
# roofline and the printed table below are the required relief for.
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = "osD^vhp<"
CEILING_VISIBLE_AT = 0.25            # draw the ceiling only once the data reaches this

_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = dict(a.split("=", 1) if "=" in a else (a, "") for a in sys.argv[1:] if a.startswith("--"))
RUN = _pos[0]
OUT = _pos[1] if len(_pos) > 1 else RUN
os.makedirs(OUT, exist_ok=True)

# Optional reference measurement: the SAME evaluator JSON, produced by running the
# problem's own reference as a solution. Without it the plots answer "how fast did the
# agent get", which is not the question anyone asks -- the question is whether it beat
# the obvious implementation. Auto-picked up from <run_dir>/reference.json so dropping
# the file next to a run is all it takes.
REF_PATH = _flags.get("--reference") or os.path.join(RUN, "reference.json")


def load_reference(path):
    """{config_name: per_config dict} from an evaluator JSON, or {} if absent.

    Tolerates the file holding a whole evaluator run's output: the JSON line is picked
    out of whatever else was captured alongside it.
    """
    if not os.path.exists(path):
        return {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "per_config" in doc:
            return {pc["name"]: pc for pc in doc["per_config"] if pc.get("name")}
    return {}


REF = load_reference(REF_PATH)


def jsonl(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip a half-written trailing line from the live run
    return rows


nodes = jsonl(os.path.join(RUN, "history/nodes.jsonl"))
manifest = json.load(open(os.path.join(RUN, "manifest.json")))
label = f"{manifest.get('problem', 'kernelguy')} {manifest.get('policy_id', '')}".strip()
t0 = min(n["timestamp_unix_ms"] for n in nodes)
timed = sorted((n for n in nodes
                if isinstance(n.get("evaluation"), dict) and "Timed" in n["evaluation"]),
               key=lambda n: n["timestamp_unix_ms"])

# ---------------------------------------------------------------- one pass ------
# attempts[name] -> [(hours, tflops, gbps)] for every timed attempt, oldest first, and
# the two device peaks. A peak is recovered as achieved/pct rather than read from the
# run: the evaluator reports `peak_tflops`/`peak_gbps` but the Rust `Timed` struct keeps
# only the per-config list, so the fractions are the sole surviving trace. Median over
# every entry, since the peak is a device constant and division is the only noise.
ATTEMPTS, peaks_t, peaks_b = {}, [], []
for n in timed:
    hours = (n["timestamp_unix_ms"] - t0) / 3.6e6
    for pc in n["evaluation"]["Timed"].get("per_config", []):
        name = pc.get("name")
        if not name:
            continue
        tf, gb = pc.get("achieved_tflops"), pc.get("achieved_gbps")
        ATTEMPTS.setdefault(name, []).append((hours, tf, gb))
        for ach, pct, bucket in ((tf, pc.get("pct_peak"), peaks_t),
                                 (gb, pc.get("pct_bandwidth"), peaks_b)):
            if ach and pct and ach > 0 and pct > 0:
                bucket.append(ach / pct)

PEAK_T = statistics.median(peaks_t) if peaks_t else None
PEAK_B = statistics.median(peaks_b) if peaks_b else None
# FLOP/byte at which the two ceilings cross. Above it a kernel cannot be fed fast
# enough to matter; below it the flat FLOP peak is unreachable by construction.
RIDGE = 1000 * PEAK_T / PEAK_B if PEAK_T and PEAK_B else None

NAMES = sorted(ATTEMPTS)
COLOR = {n: CAT[i % len(CAT)] for i, n in enumerate(NAMES)}
MARK = {n: MARKERS[i % len(MARKERS)] for i, n in enumerate(NAMES)}
HAVE_TFLOPS = any(tf for v in ATTEMPTS.values() for _, tf, _ in v)
HAVE_GBPS = any(gb for v in ATTEMPTS.values() for _, _, gb in v)


def intensity(name):
    """Median FLOP/byte over this config's attempts, or None if a metric is missing.

    Median rather than any single attempt because a problem whose `bytes_moved` depends
    on the drawn inputs (a sparse KV cache counting its distinct blocks) moves a little
    between attempts. It is otherwise constant to the digit.
    """
    vs = [1000 * tf / gb for _, tf, gb in ATTEMPTS[name] if tf and gb]
    return statistics.median(vs) if vs else None


INTENSITY = {n: intensity(n) for n in NAMES}
HAVE_SPLIT = RIDGE is not None and any(v is not None for v in INTENSITY.values())
COMPUTE = {n for n in NAMES if HAVE_SPLIT and INTENSITY[n] and INTENSITY[n] >= RIDGE}
MEMORY = {n for n in NAMES if HAVE_SPLIT and INTENSITY[n] and INTENSITY[n] < RIDGE}


def attainable_tflops(name):
    """The roof at this config's intensity: min(flat FLOP peak, what bandwidth feeds)."""
    i = INTENSITY.get(name)
    return None if not (i and PEAK_T and PEAK_B) else min(PEAK_T, i * PEAK_B / 1000)


def best_solution_steps():
    """[(hours, geomean, {name: (tflops, gbps)})] once per new CONFIRMED best.

    The record is a *solution*, not a per-config maximum: whenever a solution becomes the
    new confirmed best, its numbers become the current record for every config.

    Confirmed, not merely highest-scoring, and that distinction is quantitative. A single
    evaluation can land optimistically; the search re-measures a candidate before
    promoting it (`confirmed_score.samples`), and one that does not reproduce is never
    promoted. Stepping on raw `geomean_speedup` would draw a record no surviving solution
    achieved -- measured on the 10h TP4 run, the raw best claimed 1226 TFLOP/s at m8192
    where the promoted solution reproduces 961, so a reference ratio read off the raw
    curve overstated the win by roughly a fifth.
    """
    out, best = [], float("-inf")
    for n in timed:
        g = (n.get("confirmed_score") or {}).get("geomean")
        if g is None or g <= best:
            continue
        best = g
        vals = {pc["name"]: (pc.get("achieved_tflops"), pc.get("achieved_gbps"))
                for pc in n["evaluation"]["Timed"].get("per_config", []) if pc.get("name")}
        out.append(((n["timestamp_unix_ms"] - t0) / 3.6e6, g, vals))
    return out


STEPS = best_solution_steps()


def step_series(name, axis):
    """([hours], [value]) of the confirmed-best record for one config on one axis."""
    xs, ys = [], []
    for h, _, vals in STEPS:
        v = (vals.get(name) or (None, None))[axis]
        if v and v > 0:
            xs.append(h)
            ys.append(v)
    return xs, ys


def spread_labels(ax, items, dx=8.0):
    """Direct-label (x, y, text, colour) points, pushed apart vertically.

    Labels are placed beside their own point rather than in a right-hand column: a
    leader running the width of the axes reads as a data series. Offsets are searched
    outward from zero so an isolated point keeps a flush label and only a cluster pays.
    """
    ax.figure.canvas.draw()
    kpt = 72.0 / ax.figure.dpi
    right = ax.get_window_extent().x1 * kpt
    boxes = []
    for x, y, text, col in sorted(items, key=lambda it: -it[1]):
        px, py = ax.transData.transform((x, y))
        px, py = px * kpt, py * kpt
        w = 4.7 * len(text)                   # 4.7pt per char approximates 8pt text
        # A label on the highest-intensity config would run off the axes, so anything
        # that does not fit to the right is placed to the left of its own point instead.
        flip = px + dx + w > right
        x0 = px - dx - w if flip else px + dx
        dy = next((c for c in [0.0] + [s * 11.5 * k for k in range(1, 14) for s in (1, -1)]
                   if all(x0 + w <= q0 or q1 <= x0 or abs(py + c - qy) >= 10.5
                          for q0, q1, qy in boxes)), 0.0)
        boxes.append((x0, x0 + w, py + dy))
        ax.annotate(text, xy=(x, y), xytext=(-dx if flip else dx, dy),
                    textcoords="offset points", fontsize=8, color=LEGEND_INK,
                    va="center", ha="right" if flip else "left", zorder=7,
                    arrowprops=(dict(arrowstyle="-", color=col, lw=0.8, alpha=0.6,
                                     shrinkA=1, shrinkB=1) if abs(dy) > 6 else None))


# ---------------------------------------------------------------- roofline ------
def plot_roofline(fname):
    """Every config at its own intensity, against the roof it is actually under."""
    pts = [(n, INTENSITY[n]) for n in NAMES if INTENSITY[n]]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = min(i for _, i in pts) / 2.5
    hi = max(i for _, i in pts) * 2.5
    # The roof itself: bandwidth-limited on a slope of one until it meets the flat FLOP
    # peak. Sampled rather than drawn as two segments so the knee lands exactly on the
    # ridge at any zoom.
    xr = [lo * (hi / lo) ** (k / 240.0) for k in range(241)]
    ax.plot(xr, [min(PEAK_T, x * PEAK_B / 1000) for x in xr],
            color=PEAK_RED, ls="--", lw=1.6, zorder=2)
    ax.axvline(RIDGE, color=PEAK_RED, ls=":", lw=1.0, alpha=0.7, zorder=1)
    ax.annotate(f"ridge {RIDGE:,.0f} FLOP/byte", xy=(RIDGE, PEAK_T), xytext=(4, -14),
                textcoords="offset points", fontsize=8, color=PEAK_RED, ha="left")

    labels, table = [], []
    for name, i in pts:
        col, att = COLOR[name], attainable_tflops(name)
        ys = [tf for _, tf, _ in ATTEMPTS[name] if tf and tf > 0]
        # Intensity is fixed for the config, so every attempt stacks on one vertical and
        # the spread IS the search's range on this config. Fanned out by at most 3% --
        # deterministically, so the plot is reproducible -- purely so the column reads as
        # a band of countable attempts rather than being hidden under the arrow that
        # shares its x. Only the marker and the arrow sit on the true intensity.
        ax.scatter([i * (1 + 0.03 * ((k % 5) - 2) / 2) for k in range(len(ys))], ys,
                   s=12, alpha=0.3, color=col, linewidths=0, zorder=3)
        _, sy = step_series(name, 0)
        best = sy[-1] if sy else (max(ys) if ys else None)
        if best is None:
            continue
        first = ys[0] if ys else None
        if first and abs(best - first) / first > 0.02:
            # Vertical by construction, so the arrow's length is exactly what the search
            # bought on this config -- it cannot be confused with a change of regime. The
            # surface-coloured halo is what separates it from its own scatter column.
            arr = ax.annotate("", xy=(i, best), xytext=(i, first),
                              arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                              shrinkA=0, shrinkB=0), zorder=4)
            arr.arrow_patch.set_path_effects(
                [pe.withStroke(linewidth=3.4, foreground=SURFACE)])
        ax.scatter([i], [best], s=90, marker=MARK[name], color=col,
                   edgecolors=SURFACE, linewidths=1.0,
                   zorder=6 if sy else 5, alpha=1.0 if sy else 0.55)
        pct = best / att if att else None
        labels.append((i, best, f"{name} · {pct:.0%} of roof" if pct else name, col))
        table.append((name, i, best, att, pct, bool(sy)))
        # The problem's own reference at the same intensity: hollow, because the filled
        # marker means "what the run achieved" and this is the bar it had to clear.
        rtf = (REF.get(name) or {}).get("achieved_tflops") or 0
        if rtf > 0:
            ax.scatter([i], [rtf], s=90, marker=MARK[name], facecolors="none",
                       edgecolors=col, linewidths=1.4, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(top=PEAK_T * 1.25)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"{label}: roofline at the confirmed best "
                 f"(peaks {PEAK_T:,.0f} TFLOP/s · {PEAK_B:,.0f} GB/s; "
                 f"scatter: each attempt, arrow: first to best)")
    ax.grid(alpha=0.3, which="both")
    handles = [mlines.Line2D([], [], color=PEAK_RED, ls="--", lw=1.6)]
    keys = ["dashed: roof — min(FLOP peak, intensity x BW peak)"]
    if any((REF.get(n) or {}).get("achieved_tflops") for n, _ in pts):
        handles.append(mlines.Line2D([], [], color=LEGEND_INK, marker="o", ls="",
                                     mfc="none", mew=1.4))
        keys.append("hollow: same config's reference (torch)")
    ax.legend(handles, keys, fontsize=8, loc="lower right")
    # Lay out before labelling: spread_labels measures in display space, so a later
    # tight_layout would move the axes out from under its collision test.
    fig.tight_layout()
    spread_labels(ax, labels)
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    return table


# ------------------------------------------------------------- time series ------
def plot_timeseries(names, axis, peak, unit, peak_label, fname):
    """Achieved `unit` over wall time for `names`, one axis, ceiling where it fits.

    Writes nothing when no config has data on this axis: an empty PNG is an artifact
    someone opens and puzzles over, and the caller says why on stdout instead.
    """
    if not any(vs[axis] for n in names for _, *vs in ATTEMPTS[n]):
        return 0, 0
    fig, ax = plt.subplots(figsize=(13, 7))
    drawn, points = 0, 0
    x_end = max((h for n in names for h, _, _ in ATTEMPTS[n]), default=0.0)
    top = max((v for n in names for _, *vs in ATTEMPTS[n] if (v := vs[axis]) and v > 0),
              default=0.0)
    for name in sorted(names):
        pts = [(h, v) for h, *vs in ATTEMPTS[name] if (v := vs[axis]) and v > 0]
        if not pts:
            continue
        drawn, points = drawn + 1, points + len(pts)
        col = COLOR[name]
        sx, sy = step_series(name, axis)
        # What the problem's own reference achieves on this config, if measured. Carried
        # into the label as a ratio because that is the question the chart is asked:
        # 1.04x means the kernel beat the obvious implementation by 4%, and anything
        # below 1.00x means it lost, however large the run's own geomean looked.
        ref = (REF.get(name) or {}).get("achieved_tflops" if axis == 0 else "achieved_gbps") or 0
        if sy:
            of_peak = f", {sy[-1] / peak:.0%} of peak" if peak else ""
            gain = f", {sy[-1] / ref:.2f}x ref" if ref else ""
            ax.step([*sx, x_end], [*sy, sy[-1]], where="post", lw=2, color=col,
                    label=f"{name} — {sy[-1]:,.0f} {unit} at confirmed best{of_peak}{gain}")
        else:
            ax.plot([], [], lw=2, color=col, label=f"{name} — never in a confirmed best")
        # Every timed attempt as a scatter behind it, so the spread and the attempt
        # count stay readable without competing with the record curve.
        ax.scatter([h for h, _ in pts], [v for _, v in pts], s=14, alpha=0.35,
                   color=col, linewidths=0)
        # The reference in the config's OWN colour: colour already means config identity,
        # so the reference needs no hue of its own, only a line style. Dotted, because
        # dashed is the device peak.
        if ref:
            ax.hlines(ref, 0, x_end, color=col, ls=":", lw=1.4, alpha=0.95)
    # A ceiling far above the data compresses every curve into a strip at the bottom, so
    # it is drawn only once the run is within reach of it. Otherwise the number goes in
    # the title and each label's own fraction-of-peak carries the headroom instead.
    headroom = f"peak {peak:,.0f} {unit}" if peak else "device peak unknown"
    if peak and top >= CEILING_VISIBLE_AT * peak:
        ax.axhline(peak, color=PEAK_RED, ls="--", lw=1.4)
        ax.set_ylim(top=max(ax.get_ylim()[1], peak * 1.03))
        lo, hi = ax.get_ylim()
        keep = [t for t in ax.get_yticks() if lo <= t <= hi and abs(t - peak) > 0.045 * (hi - lo)]
        ax.set_yticks([*keep, peak])
        ax.set_yticklabels([*(f"{t:,.0f}" for t in keep), f"{peak:,.0f}\n{peak_label}"])
        ax.get_yticklabels()[-1].set(color=PEAK_RED, fontweight="bold")
        headroom = ""
    elif peak and top:
        headroom = f"peak {peak:,.0f} {unit} off-scale — top attempt is {top / peak:.1%} of it"
    ax.set_xlabel("wall time (hours)")
    ax.set_ylabel(f"achieved {unit}")
    ax.set_title(f"{label}: {unit} of the best solution so far (scatter: each attempt)"
                 + (f"\n{headroom}" if headroom else ""))
    ax.grid(alpha=0.3, which="both")
    if drawn:
        # matplotlib warns on a legend with nothing to label, which a run with no
        # timed attempts yet would otherwise hit on every invocation.
        handles, labels_ = ax.get_legend_handles_labels()
        if any((REF.get(n) or {}).get("achieved_tflops" if axis == 0 else "achieved_gbps")
               for n in names):
            handles.append(mlines.Line2D([], [], color=LEGEND_INK, ls=":", lw=1.4))
            labels_.append("dotted: same config's reference (torch)")
        ax.legend(handles, labels_, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    return drawn, points


# ------------------------------------------------------------------ drive ------
def main():
    if not NAMES:
        print("no timed per-config data in this run yet")
        return
    if HAVE_SPLIT:
        table = plot_roofline("viz_roofline.png")
        print(f"peaks: {PEAK_T:,.0f} TFLOP/s · {PEAK_B:,.0f} GB/s   "
              f"ridge {RIDGE:,.1f} FLOP/byte")
        print(f"  {'config':26s} {'FLOP/byte':>10s} {'bound by':>9s} "
              f"{'roof':>10s} {'best':>10s} {'of roof':>8s}")
        for name, i, best, att, pct, promoted in sorted(table, key=lambda r: -(r[4] or 0)):
            print(f"  {name:26s} {i:10.2f} {'compute' if name in COMPUTE else 'memory':>9s} "
                  f"{att:9,.0f}TF {best:9,.1f}TF {(f'{pct:7.1%}' if pct else ' ' * 8)}"
                  + ("" if promoted else "   (best attempt; never promoted)"))
        views = [(COMPUTE, 0, PEAK_T, "TFLOP/s", "FLOP peak", "viz_compute_tflops.png",
                  f"all {len(NAMES)} configs sit below the ridge, so none is compute-bound"),
                 (MEMORY, 1, PEAK_B, "GB/s", "BW peak", "viz_memory_bw.png",
                  f"all {len(NAMES)} configs sit above the ridge, so none is memory-bound")]
    else:
        # No ridge, so nothing can be classified. Draw whichever axes have data over
        # every config and name the reason, rather than emitting two honest-looking
        # empty plots.
        # Missing metric before missing peak: a problem defining only one of the two has
        # no intensity whatever the device tables say, so blaming the tables would send
        # the reader to the wrong file.
        why = ("the problem defines only one of flops()/bytes_moved(), so no config has "
               "an arithmetic intensity"
               if not (HAVE_TFLOPS and HAVE_GBPS) else
               "device peaks unknown — this device is missing from problem_loader.py's "
               "tables, so the evaluator recorded no pct_peak/pct_bandwidth to derive "
               "them from")
        print(f"no roofline: {why}")
        views = [(set(NAMES), 0, PEAK_T, "TFLOP/s", "FLOP peak", "viz_compute_tflops.png",
                  "no achieved_tflops recorded: the problem defines no flops()"),
                 (set(NAMES), 1, PEAK_B, "GB/s", "BW peak", "viz_memory_bw.png",
                  "no achieved_gbps recorded: the problem defines no bytes_moved()")]
    for names, axis, peak, unit, peak_label, fname, note in views:
        n_cfg, n_pts = plot_timeseries(names, axis, peak, unit, peak_label, fname)
        print(f"{unit:>8s}: {n_cfg} configs, {n_pts} points -> {os.path.join(OUT, fname)}"
              if n_cfg else f"{unit:>8s}: no plot — {note}")


main()
