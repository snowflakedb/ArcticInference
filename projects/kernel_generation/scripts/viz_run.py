#!/usr/bin/env python3
"""Throwaway viz for a kernelguy run dir. Usage: viz_run.py <run_dir>
Emits <run_dir>/viz_score.png, viz_tree.png, viz_timeline.png, viz_branches.png and a
text summary. The eval-only performance tree lives in its own viz_tree_perf.py."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = sys.argv[1]
def jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]

nodes = jsonl(os.path.join(RUN, "history/nodes.jsonl"))
manifest = json.load(open(os.path.join(RUN, "manifest.json")))
state = json.load(open(os.path.join(RUN, "history/state.json")))
by_id = {n["node_id"]: n for n in nodes}
LABEL = f"{manifest.get('problem', 'kernelguy')} {manifest.get('policy_id', '')}".strip()

def kind_and_geo(n):
    ev = n.get("evaluation")
    if not isinstance(ev, dict): return ("msg", None, None)      # message-only / unscored
    if "Timed" in ev:   return ("timed", ev["Timed"]["geomean_speedup"], ev["Timed"].get("stage"))
    if "Verified" in ev: return ("verified", None, ev["Verified"].get("stage"))
    if "Failed" in ev:  return ("failed", None, ev["Failed"].get("stage"))
    return ("msg", None, None)

COLORS = {"timed": "#2a9d3f", "verified": "#e0a020", "failed": "#c0392b", "msg": "#b8c0cc"}
t0 = min(n["timestamp_unix_ms"] for n in nodes)
def secs(n): return (n["timestamp_unix_ms"] - t0) / 1000.0

confirmed = {n["node_id"]: n["confirmed_score"]["geomean"]
             for n in nodes if n.get("confirmed_score")}
best_id = manifest.get("best_node_id")

# ---------- text summary ----------
counts = {}
for n in nodes:
    counts[kind_and_geo(n)[0]] = counts.get(kind_and_geo(n)[0], 0) + 1
timed = [(n["node_id"], kind_and_geo(n)[1]) for n in nodes if kind_and_geo(n)[0] == "timed"]
led = manifest.get("ledger", {})
print(f"== run {os.path.basename(RUN)} ==")
print(f"nodes: {len(nodes)}  kinds: {counts}")
print(f"turns={led.get('turns')} evals={led.get('evaluations')} wall={led.get('wall_clock_secs')}s "
      f"out_tok={led.get('output_tokens')} commits={led.get('commits')}")
print(f"timed candidates: {len(timed)}  -> geomeans: {[round(g,4) for _,g in timed]}")
print(f"confirmed nodes: {[(k[:8], round(v,4)) for k,v in confirmed.items()]}")
print(f"best: {manifest.get('best_geomean_speedup')} (node {str(best_id)[:8]})")
print(f"frontier size: {len(state.get('frontier', []))}")

# ---------- Plot 1: score / candidates over time ----------
fig, ax = plt.subplots(figsize=(9, 5))
curve = manifest.get("curve", [])
if curve:
    xs = [c.get("wall_clock_secs", 0) for c in curve]
    ys = [c.get("best_geomean", 0) for c in curve]
    ax.step(xs, ys, where="post", color="#1f4e8c", lw=2, label="confirmed best")
    ax.scatter(xs, ys, color="#1f4e8c", zorder=3)
# all timed candidates as points at their creation time
cand_x = [secs(n) for n in nodes if kind_and_geo(n)[0] == "timed"]
cand_y = [kind_and_geo(n)[1] for n in nodes if kind_and_geo(n)[0] == "timed"]
ax.scatter(cand_x, cand_y, color="#2a9d3f", alpha=0.6, s=40, label="timed candidate")
ax.axhline(1.0, color="#888", ls="--", lw=1, label="baseline (1.0x)")
ax.set_xlabel("wall seconds"); ax.set_ylabel("geomean speedup vs baseline")
ax.set_title(f"{LABEL}: score over time  ({os.path.basename(RUN)})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RUN, "viz_score.png"), dpi=110); plt.close(fig)

# ---------- Plot 2: search tree ----------
# layered layout: x = turn, y = lane (assigned to avoid sibling overlap)
children = {}
for n in nodes:
    children.setdefault(n.get("parent_id"), []).append(n["node_id"])
lane = {}
nxt = [0]
def assign(nid, depth):
    kids = sorted(children.get(nid, []), key=lambda k: by_id[k]["timestamp_unix_ms"])
    if not kids:
        lane[nid] = nxt[0]; nxt[0] += 1; return lane[nid]
    ls = [assign(k, depth+1) for k in kids]
    lane[nid] = sum(ls)/len(ls)
    return lane[nid]
roots = [n["node_id"] for n in nodes if not n.get("parent_id")]
for r in sorted(roots, key=lambda k: by_id[k]["timestamp_unix_ms"]):
    assign(r, 0)
for n in nodes:
    lane.setdefault(n["node_id"], nxt[0]);
    if n["node_id"] not in lane: nxt[0]+=1

fig, ax = plt.subplots(figsize=(16, 9))
for n in nodes:
    pid = n.get("parent_id")
    if pid and pid in lane:
        ax.plot([by_id[pid]["turn"], n["turn"]], [lane[pid], lane[n["node_id"]]],
                color="#ccc", lw=0.8, zorder=1)
for n in nodes:
    k, g, _ = kind_and_geo(n)
    nid = n["node_id"]
    # Marker size grows with speedup but is CAPPED so high-speedup nodes (20x+)
    # don't blow up into overlapping blobs.
    size = 60 if k != "timed" else min(360.0, 60 + 30 * max(0.0, (g or 1) - 0.9))
    ax.scatter(n["turn"], lane[nid], s=size, color=COLORS[k],
               edgecolors=("k" if nid in confirmed else "none"),
               linewidths=1.6, zorder=2)
    # Only label the committed lineage (confirmed nodes) — labeling every timed
    # candidate collides into an unreadable mass at scale.
    if nid in confirmed:
        ax.annotate(f"{confirmed[nid]:.2f}x", (n["turn"], lane[nid]), fontsize=7,
                    xytext=(3, 4), textcoords="offset points", zorder=4)
    if nid == best_id:
        ax.scatter(n["turn"], lane[nid], s=size+180, facecolors="none",
                   edgecolors="#d62728", linewidths=2.2, zorder=1)
handles = [plt.Line2D([0],[0], marker='o', ls='', mfc=c, mec='none', label=k) for k,c in COLORS.items()]
handles.append(plt.Line2D([0],[0], marker='o', ls='', mfc='none', mec='k', label='confirmed'))
handles.append(plt.Line2D([0],[0], marker='o', ls='', mfc='none', mec='#d62728', label='best'))
ax.legend(handles=handles, loc="upper left", fontsize=8)
ax.set_xlabel("turn"); ax.set_ylabel("branch lane")
ax.set_title(f"{LABEL}: search tree  (nodes={len(nodes)}, timed={len(timed)}, frontier={len(state.get('frontier',[]))})")
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(RUN, "viz_tree.png"), dpi=130); plt.close(fig)

# ---------- Plot 3: node-creation timeline (concurrency) ----------
# Group nodes into episodes by walking parent chains; each maximal chain of
# same-branch appends is a lane. Simpler proxy: color by kind, y = lane from tree,
# x = creation time. Concurrent expanders show as overlapping x-spans on different lanes.
fig, ax = plt.subplots(figsize=(11, 5))
for n in nodes:
    pid = n.get("parent_id")
    if pid and pid in lane:
        ax.plot([secs(by_id[pid]), secs(n)], [lane[pid], lane[n["node_id"]]],
                color="#ddd", lw=0.7, zorder=1)
for n in nodes:
    k, g, _ = kind_and_geo(n)
    ax.scatter(secs(n), lane[n["node_id"]], s=50, color=COLORS[k], zorder=2)
ax.set_xlabel("wall seconds"); ax.set_ylabel("branch lane")
ax.set_title(f"{LABEL}: node creation timeline (overlapping lanes = concurrent expanders)")
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(RUN, "viz_timeline.png"), dpi=110); plt.close(fig)

# ---------- shared derivation for Plot 5 (the tree plot moved to viz_tree_perf.py) ----------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXISC = "#e1e0d9", "#c3c2b7"
EVK = {n["node_id"]: kind_and_geo(n)[0] for n in nodes if kind_and_geo(n)[0] != "msg"}
GEO = {nid: kind_and_geo(by_id[nid])[1] for nid, k in EVK.items() if k == "timed"}

# One pass: carry each timed score down to descendants, and hang every evaluation off
# the nearest evaluation strictly above it.
score, cparent = {}, {}
st = [(r, None, None) for r in roots]
while st:
    nid, carried, near = st.pop()
    cur = GEO.get(nid, carried)
    score[nid] = cur
    nxt_near = nid if nid in EVK else near
    if nid in EVK:
        cparent[nid] = near
    st.extend((k, cur, nxt_near) for k in children.get(nid, []))

hrs = {e: secs(by_id[e]) / 3600.0 for e in cparent}
span = max(hrs.values(), default=0.0) or 1.0

# ---------- Plot 5: per-branch performance trajectory ----------
# y is the thing being optimised, so branches that overtake each other actually cross,
# and each branch keeps one identity across the whole run. A branch here is a heavy-path
# chain of the contracted eval tree: at every fork the largest subtree continues the
# branch and the smaller ones start new branches, so each evaluation belongs to exactly
# one branch and no two branches share a segment.
csize, ckids = {}, {e: [] for e in cparent}
for c, p in cparent.items():
    if p is not None:
        ckids[p].append(c)
for v in ckids.values():
    v.sort(key=lambda k: by_id[k]["timestamp_unix_ms"])
croots = [c for c, p in cparent.items() if p is None]
for r in croots:                                          # iterative post-order sizes
    post, stk = [], [r]
    while stk:
        nid = stk.pop(); post.append(nid); stk.extend(ckids[nid])
    for nid in reversed(post):
        csize[nid] = 1 + sum(csize[k] for k in ckids[nid])
bid, nbr = {}, 0
for r in croots:
    bid[r] = nbr; nbr += 1
    stk = [r]
    while stk:
        nid = stk.pop()
        for i, k in enumerate(sorted(ckids[nid], key=lambda k: (-csize[k], by_id[k]["timestamp_unix_ms"]))):
            bid[k] = bid[nid] if i == 0 else nbr
            nbr += 0 if i == 0 else 1
            stk.append(k)

seq = {}
for e in cparent:
    if EVK[e] == "timed":
        seq.setdefault(bid[e], []).append(e)
for v in seq.values():
    v.sort(key=lambda e: hrs[e])
# Order by peak so the branches crowding the high-speedup band all get a solid distinct
# hue, which is where telling one line from another actually matters. Peak is fixed once
# the run is over, so this is a stable property of the branch, not a live ranking.
order = sorted(seq, key=lambda b: -max(GEO[e] for e in seq[b]))
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# 16 branches on 8 hues: hue x {solid, dashed} gives each one a unique pair. Required
# anyway -- all-pairs CVD bottoms out at dE 11.2 (orange/green, protan), which is legal
# only alongside secondary encoding, and three hues sit under 3:1 on this surface. The
# dash channel plus the direct end-labels below are that relief.
style = {b: (CAT[i % len(CAT)], "-" if i < len(CAT) else (0, (6, 2.5)),
             "osD^vhp<"[i % len(CAT)]) for i, b in enumerate(order)}

fig, ax = plt.subplots(figsize=(15, 9), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for b in order:
    col, dash, mk = style[b]
    xs = [hrs[e] for e in seq[b]]
    ys = [GEO[e] for e in seq[b]]
    par = cparent[seq[b][0]]                              # show where the branch forked
    if par is not None and score[par] is not None:
        ax.plot([hrs[par], xs[0]], [score[par], ys[0]], color=col, ls=dash, lw=0.9,
                alpha=0.3, zorder=2)
    ax.plot(xs, ys, color=col, ls=dash, lw=1.6, solid_capstyle="round", zorder=3)
    ax.scatter(xs, ys, s=24, marker=mk, color=col, edgecolors=SURFACE, linewidths=0.8, zorder=4)
if best_id in GEO:
    ax.scatter([hrs[best_id]], [GEO[best_id]], s=300, facecolors="none", edgecolors=INK,
               linewidths=1.8, zorder=6)

ylo, yhi = 0.0, (max(GEO.values()) * 1.06 if GEO else 2.0)
ax.set_xlim(-0.15, span * 1.08)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("wall-clock hours", color=INK2, fontsize=10)
ax.set_ylabel("geomean speedup vs baseline", color=INK2, fontsize=10)
ax.set_title(f"{LABEL} · {len(seq)} branches, {len(GEO)} timed evaluations  "
             f"({os.path.basename(RUN.rstrip('/'))})", color=INK, fontsize=12, pad=14, loc="left")
ax.grid(axis="y", color=GRIDC, lw=0.8, ls="-")            # y carries the values
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXISC)
ax.tick_params(colors=MUTED, labelsize=9)
# No legend: every branch is labelled at its own line end, so a 16-entry box would
# restate all 16 and nothing else. The direct labels are the identity channel.
fig.tight_layout()

# Each branch is labelled at its OWN last point, not in a far-right column: a branch that
# died at 3h would otherwise get a leader running the full width of the chart, which reads
# as a data series. Labels start beside the endpoint and are pushed vertically only until
# they stop overlapping, so the converging top cluster forms a short local column and
# every early-dying branch is labelled in place. Direct labels are mandatory here, not
# decorative -- all-pairs CVD bottoms out at dE 11.2 and three hues sit under 3:1.
fig.canvas.draw()
kpt = 72.0 / fig.dpi
boxes = []
for b in sorted(order, key=lambda b: -GEO[seq[b][-1]]):
    e = seq[b][-1]
    px, py = ax.transData.transform((hrs[e], GEO[e]))
    px, py = px * kpt, py * kpt
    txt = f"b{b} · {GEO[e]:.0f}x"
    w = 4.7 * len(txt)
    x0 = px + 10.0
    dy = 0.0
    for cand in [0.0] + [s * 11.5 * n for n in range(1, 14) for s in (1, -1)]:
        if all(x0 + w <= qx0 or qx1 <= x0 or abs(py + cand - qy) >= 10.5
               for qx0, qx1, qy in boxes):
            dy = cand
            break
    boxes.append((x0, x0 + w, py + dy))
    col = style[b][0]
    ax.annotate(txt, xy=(hrs[e], GEO[e]), xycoords="data", xytext=(10.0, dy),
                textcoords="offset points", fontsize=8, color=INK2, va="center",
                ha="left", zorder=7,
                arrowprops=(dict(arrowstyle="-", color=col, lw=0.8, alpha=0.6,
                                 shrinkA=1, shrinkB=1) if abs(dy) > 6 else None))

fig.savefig(os.path.join(RUN, "viz_branches.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

nodraw = nbr - len(seq)
print(f"branches: {len(seq)} drawn on {len(CAT)} hues x solid/dashed"
      + (f", {nodraw} not drawn (no timed eval, so no y)" if nodraw else ""))
print(f"  {'br':>3} {'evals':>5} {'peak':>7} {'first_h':>7} {'last_h':>7}")
for b in order:
    print(f"  {b:3d} {len(seq[b]):5d} {max(GEO[e] for e in seq[b]):6.1f}x "
          f"{hrs[seq[b][0]]:7.2f} {hrs[seq[b][-1]]:7.2f}")
print("wrote viz_score.png, viz_tree.png, viz_timeline.png, viz_branches.png")
