# Cross-node restore with `SEMIP_UNPRIVILEGED=1`

Runbook for dumping a semi-persistence image on node A and restoring it on a
different node B that has only `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`.

Background for every claim here: *Complication 11* and *Complication 7* in
[`CRIU_PLUMBING.md`](CRIU_PLUMBING.md), and *Cross-node restore* in
[`semi-p_DESIGN.md`](semi-p_DESIGN.md). Node qualification is in
[`INSTALL.md`](INSTALL.md).

---

## 0. The one thing that cannot be fixed later

`SEMIP_UNPRIVILEGED=1` must be set **on the dump side too**, not just on the
restore side. It is one switch with two jobs:

- On dump it adds `--unprivileged` to `criu dump` *and* makes the vLLM child
  zero its capabilities before `import torch`, so CRIU records an empty cap set
  for every task in the tree.
- On restore it selects `_worker_criu_load_lowcap` (`criu restore -d
  --unprivileged`, no `unshare`, no private `/proc`).

CRIU's `restore_creds()` reinstates whatever caps the image recorded. An image
dumped **without** the flag records real capabilities that node B cannot grant,
and dies in `criu/pie/restorer.c` with *"Unable to restore capabilities"*. There
is no way to downgrade an existing image — it has to be re-dumped.

So set it before the dump, in the driver process, before `Instance` is used:

```python
import os
os.environ["SEMIP_UNPRIVILEGED"] = "1"
```

and set it again in the restore driver on node B.

---

## 1. Dump side (node A)

```bash
cd /code/users/mert/ArcticInference/arctic_inference/semi_persistence
```

Give the instance an explicit `model_dir` so the image, the weights and the
compile cache land under one tree that can be copied as a unit:

```python
inst = Instance(vllm_config, "/data-fast/image-cache/qwen_35b")
#   <model_dir>/image        CRIU image + meta.json
#   <model_dir>/weights      only if you call save_weights()
#   <model_dir>/compilation  vLLM compile cache (mmapped .so/cubin by abs path)
```

Three rules while dumping:

- **Note which GPU indices you dumped on.** You need to restore onto *different*
  ones (see §4).
- **Do not rebuild the C extensions after the dump.** The image records
  `arctic_inference/*.so` by path and size; rebuilding changes the bytes and
  invalidates it. The repo is on shared Lustre, so a rebuild on *either* node
  breaks images captured by both.
- **Put the tree high in the PID space.** CRIU recreates every task at its
  recorded id and node B must have all of them free (§5), but a container hands
  out low ids to whatever starts first — so a tree captured around PID 1200
  competes forever with node B's own long-lived services. Advance the counter
  before the instance exists with `python3 scripts/pidcheck.py --burn-to
  200000` (namespace-global, so it need not be the same shell). Ids that high
  are unreachable in normal operation — `pid_max` is 4194304 — which retires
  the whole collision class for that image. It is also the *only* fix when the
  occupant on node B turns out to be your own login shell or IDE server, which
  you cannot kill and cannot outrun.

---

## 2. Preflight on node B (before copying anything)

```bash
# a. CRIU 4.2 with the CUDA plugin, and passwordless sudo
criu --version                        # 4.2.x
ls /usr/lib/criu/cuda_plugin.so
sudo -n true && echo "sudo OK"

# b. The kernel-feature probe in the mode the restore actually uses.
#    Plain `criu check` aborts here on a low-cap node; --unprivileged is the
#    real gate. A residual read-only `ns_last_pid` complaint is expected and
#    does NOT block restore.
sudo criu check --unprivileged

# c. Same GPU count and model as node A (the cuda-checkpoint device map is a
#    bijection over EVERY visible GPU, so the counts must match).
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

`/usr/lib/criu/empty` is only needed by `criu dump`, which creates it itself.
Restore never passes `--libdir`, so node B does not need it.

### The cap-drop precondition (do not skip this on the restore node)

After the cap drop, uid 0 loses `CAP_DAC_OVERRIDE` and can only read what
`other` can. `criu_restore` itself imports nothing and is safe, but **anything
that runs after it may not be** — at TP>1 `reinit_nccl` reaches vLLM's
`in_the_same_node_as`, which spawns a fresh interpreter via
`multiprocessing.resource_tracker`. If that spawn cannot read its own stdlib,
it dies, `BrokenPipeError` is swallowed by vLLM's `contextlib.suppress(OSError)`,
and the restore **hangs silently** with only `No available shared memory
broadcast block found in 60 seconds` in the log.

```bash
# Every component needs o+x, files o+r. Check the BASE prefix: a venv shares
# its base interpreter's stdlib. A 750 home is the usual culprit.
namei -l "$(python3 -c 'import sys; print(sys.base_prefix)')"

# Functional version — makes the exact call that breaks. Must print OK.
sudo python3 -c '
import ctypes
from multiprocessing import shared_memory, resource_tracker
libc = ctypes.CDLL("libc.so.6", use_errno=True)
for c in range(64): libc.prctl(24, c, 0, 0, 0)
libc.prctl(47, 4, 0, 0, 0)
class H(ctypes.Structure): _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class D(ctypes.Structure): _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
libc.capset(ctypes.byref(H(0x20080522,0)), ctypes.byref((D*2)()))
s = shared_memory.SharedMemory(create=True, size=128); print("OK", s.name)
s.close(); s.unlink()'
```

`BrokenPipeError` instead of `OK` means fix the permissions (`chmod o+x` the
home, or serve with a world-readable prefix). No re-dump is needed — nothing
about the image is wrong.

### `arctic_inference` must be importable **as root**

The whole tree runs as root (`mp.spawn` inherits the uid; `criu` and
`cuda-checkpoint` go through `sudo`), so a `pip install --user` is invisible to
it. Verify on node B:

```bash
sudo python3 -c "from arctic_inference.semi_persistence import Instance; print('OK')"
```

If it fails, register the *same* checkout without rebuilding anything — copy the
three editable-install artifacts from node A rather than running `pip install -e .`,
which would recompile the `.so` files in the shared repo and invalidate every
existing image:

```bash
D=/usr/local/lib/python3.12/dist-packages
sudo cp -a $D/__editable__.arctic_inference-0.3.1.dev0.pth \
           $D/__editable___arctic_inference_0_3_1_dev0_finder.py \
           $D/arctic_inference-0.3.1.dev0.dist-info  <node-B>:$D/
```

The finder maps to absolute paths under `/code/users/mert/ArcticInference`,
which is shared Lustre, so node B resolves the identical files.

### …and check that nothing shadows the checkout

If node B has a *real* `dist-packages/arctic_inference/` directory — a plain
`pip install` rather than `-e`, or a hand-copied tree — that wins over the
editable finder, because `PathFinder` is consulted before the finder the `.pth`
appends to `sys.meta_path`. Every edit you make to the checkout is then inert
at runtime, with nothing to show for it but the file paths in tracebacks:

```bash
sudo python3 -c "import arctic_inference.semi_persistence as sp; print(sp.__file__)"
```

If that prints a path under `dist-packages` that is not a symlink, point it at
the checkout instead of syncing files by hand:

```bash
sudo mv $D/arctic_inference/semi_persistence $D/arctic_inference/semi_persistence.bak
sudo ln -sfn /code/users/mert/ArcticInference/arctic_inference/semi_persistence \
             $D/arctic_inference/semi_persistence
```

A symlink is safe where `pip install -e .` is not: it rebuilds nothing, so it
cannot change a recorded `.so` and cannot invalidate an existing image. Note
`scripts/` and `skills/` are not part of an installed copy, so `imgdiff.py` and
`pidcheck.py` are always run from the checkout regardless.

---

## 3. Copy the image tree to the identical absolute path

`/data-fast` is node-local XFS — the image cache does **not** travel with the
Lustre repo and must be copied. The path must match byte-for-byte, because the
image bakes the compile-cache mappings by absolute path *and* `criu_restore`
rejects a `model_dir` that differs from the one in `meta.json`.

```bash
sudo rsync -aH --info=progress2 \
    /data-fast/image-cache/qwen_35b/ \
    <node-B>:/data-fast/image-cache/qwen_35b/
```

Use `rsync -a` or a plain tarball. **Never `tar -h`** — dereferencing symlinks
changes file sizes and CRIU size-checks every mapping.

### Then verify the environment before spending a restore attempt

CRIU records every file-backed mapping by absolute path and size and
re-validates the size at restore. One mapped file of a different length aborts
the whole restore from inside CRIU, with no up-front check, and it stops at the
first bad mapping instead of reporting them all. "Same requirements installed"
is *not* sufficient — identical version specs routinely yield different bytes.

```bash
# On node B:
cd /code/users/mert/ArcticInference/arctic_inference/semi_persistence
sudo python3 scripts/imgdiff.py /data-fast/image-cache/qwen_35b/image
```

Want `TOTAL PROBLEMS: 0`. It also compares the ELF build-IDs CRIU recorded, so
it catches a same-size-different-build library that would pass CRIU's own check
and then map the wrong text pages. The one entry it always lists and that is
never a problem is the `/dev/shm/sem.*` **GHOST** file — unlinked at dump time,
carried inside the image, recreated by CRIU.

If it reports `SIZE MISMATCH` / `ABSENT` / `BUILD-ID MISMATCH`, it prints the
trees needing sync. Copy those trees from node A to the identical path (an HF
cache under `/root/.cache/huggingface` is a common one if vLLM still had
safetensors mapped at dump time) and re-run until clean.

### Then check the recorded task ids are free — immediately before restoring

```bash
sudo python3 scripts/pidcheck.py /data-fast/image-cache/qwen_35b/image
```

Without a private PID namespace this mode needs *every* recorded task id free,
threads included (§5). Unlike `imgdiff.py` this is a snapshot, not a property
of the image: the answer changes as processes come and go, so run it right
before the restore. `criu_restore` preflights the same check itself and names
the occupant, so this only buys you the answer before spending an attempt.

`--burn` advances the PID counter past the image's highest id so *subsequently
started* processes land clear of it. It frees nothing. Two consequences worth
internalising:

- A process already running keeps its ids. If `pidcheck.py` lists a live
  occupant, burning changes nothing for it — it has to exit.
- If the occupant is an ancestor of your launcher (your shell, `sudo`, the IDE
  server that spawned the terminal), you cannot outrun it either: a burn moves
  only descendants you start afterwards. `criu_restore`'s report flags this case
  as `an ancestor of this restore`. Re-dump per §1 instead.

---

## 4. Restore script for node B

```python
import json, os

os.environ["SEMIP_UNPRIVILEGED"] = "1"          # before Instance is used

from arctic_inference.semi_persistence import Instance

MODEL_DIR = "/data-fast/image-cache/qwen_35b"

# criu_restore compares vllm_config with a full dict !=, so reading it back out
# of the image is the only way to be sure the check passes.
meta = json.load(open(os.path.join(MODEL_DIR, "image", "meta.json")))
print("dumped on GPUs", meta["gpus"], "of", len(meta["gpu_uuids"]), "visible")

inst = Instance(meta["vllm_config"], MODEL_DIR)

inst.criu_restore()                  # -> state 'checkpointed'
inst.attach()                        # only if the image was dumped with
inst.load_weights()                  #   save_weights()/detach(); else drop both

inst.cuda_restore(gpus=[4, 5])       # MUST differ from meta["gpus"] -- see below
inst.reinit_nccl()                   # TP>1 only, immediately after cuda_restore
inst.wake_up_weights()
inst.repin()
inst.restore_weights()
inst.wake_up_kv_cache()
inst.recapture_graphs("reuse")       # TP>1 only, after wake_up_kv_cache
inst.generate(["Hello, world!"], {"temperature": 0.0, "max_tokens": 64})
inst.wait().print_status()
inst.teardown()
```

Run it as root, from this directory (modules import their siblings by bare
name):

```bash
cd /code/users/mert/ArcticInference/arctic_inference/semi_persistence
sudo python3 restore_here.py       # the script sets SEMIP_UNPRIVILEGED itself
```

At TP=1 use `inst.cuda_restore(gpu=3)` and drop `reinit_nccl` /
`recapture_graphs` (they are no-ops anyway).

### Restore onto GPU indices that differ from the dump's

This is the cross-node-specific trap. `_worker_restore` builds the
`oldUuid=newUuid` device map **only** when the placement changed:

```python
migrate = bool(old_gpus and new_gpus and list(old_gpus) != list(new_gpus))
```

`meta.json` carries node A's GPU UUIDs precisely so the map can pair them
against node B's locally-read UUIDs. Restore onto the *same* indices and no map
is built at all, leaving the checkpoint's baked node-A UUIDs — which do not
exist on node B — and `cuCheckpointProcessRestore` fails with
`CUDA_ERROR_INVALID_VALUE (CUresult=1)`. Zero overlap between the two UUID sets
is the normal case, not an error.

The permutation pins `old_gpus[k] -> new_gpus[k]` and pairs the remaining GPUs
in order, so it works even when the index sets overlap (`[4,5,6,7] -> [0,1,4,5]`).

---

## 5. Constraints that stay true in this mode

| Constraint | Consequence |
|---|---|
| **Every recorded task id must be free** | No private PID namespace, so CRIU recreates each task at its recorded id — leaders *and* their threads, which come from one number space (a TP2 image needs ~900 ids across 3 leaders). Any live process whose threads overlap the range blocks the restore, so at most one live restore per node, and `scripts/test_weights.py`, which restores two models concurrently, is incompatible with this mode. `criu_restore` preflights it and names the occupants; `scripts/pidcheck.py` answers it before you spend an attempt. |
| **`vllm_config` must match exactly** | Full dict `!=`, no extra keys. `tensor_parallel_size` participates, so a TP=2 image cannot be mistaken for a TP=1 one. Load it from `meta.json`. |
| **`model_dir` must match** | Rejected by name up front. |
| **Same visible GPU count** | The device map is a bijection over every visible GPU. |
| **The restored child runs the code frozen in the image** | Editing `vllm_child.py` changes nothing until an offline re-dump. Keep dump-time and runtime code the same checkout. |
| **`criu_restore` does not re-apply `_env`** | The child's environment is baked into the image and restored verbatim. |
| **A dump is destructive** | After `criu_dump` the model is `saved` with no live process. |
| **Don't restore within 60s of a dump on the same node** | Recorded local ports sit in `TIME_WAIT`. Current dumps set `SO_LINGER(1,0)` so the kill RSTs instead; older images need the 60s to drain. Cross-node this does not apply. |

---

## 6. Symptom → cause

| What you see | Cause |
|---|---|
| `Unable to restore capabilities` (`criu/pie/restorer.c`) | Image dumped **without** `SEMIP_UNPRIVILEGED=1`. Re-dump. |
| `Could not initialize kernel features detection` | `--unprivileged` not in play: `SEMIP_UNPRIVILEGED` unset in the restore driver. |
| `File <path> has bad size <local> (expect <image>)` + `Can't open vma` | Environment mismatch. Run `imgdiff.py`, sync the reported trees. |
| `CUDA_ERROR_INVALID_VALUE (CUresult=1)` from `cuda_restore` | No device map built — you restored onto the dump's own GPU indices (§4), or `meta.json` predates `gpu_uuids`. |
| `Can't bind inet socket … Address already in use` | Recorded port in `TIME_WAIT`; wait 60s. |
| `criu restore aborted (recorded task-id collision: …)` | The preflight found recorded ids occupied; the message names every occupying process, and flags one that is `an ancestor of this restore`. Free them, or `scripts/pidcheck.py <image> --burn` and relaunch, or re-dump per §1. |
| CRIU `Can't fork for <pid>: File exists`, or `pie: Unable to create a thread: -17` | The same collision, claimed in the window after the preflight ran (criu's own helpers spawn into the same number space). `-17` is `EEXIST` on a *thread* id, which is why nothing shows at that number in `ps` — threads live only under `/proc/<pid>/task`. The error appends the resolved occupants. |
| Silent hang in `reinit_nccl`; log repeats `No available shared memory broadcast block found in 60 seconds` | Cap-drop precondition violated on the restore node. `py-spy dump` shows ranks at *different* lines of `in_the_same_node_as`. Fix permissions; no re-dump. |
| `ModuleNotFoundError` for a stdlib module right after `[semip] dropped capabilities` | Same precondition, hit at cold start instead. |
| `ModuleNotFoundError: No module named 'arctic_inference'` under `sudo` | Installed `--user`; root cannot see `~/.local`. |
