# Installation

Setup steps for running `semi_persistence` on a fresh Ubuntu 24.04 host:
install CRIU (with the CUDA plugin path), then download the speculative
draft models from S3 to `/data-fast/`.

For background on *why* the CRIU plumbing looks the way it does, see
[`CRIU_PLUMBING.md`](./CRIU_PLUMBING.md).

---

## 1. CRIU (Ubuntu 24.04)

CRIU is not in the default Ubuntu repos at a recent-enough version.
Install from the official CRIU PPA:

```bash
# 0. Refresh the package lists first.  On a container whose lists are
#    stale, step 1 fails with 404s on every package because the mirrors
#    have already rotated out the versions the old lists point at.
sudo apt-get update

# 1. Add the CRIU PPA
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update

# 2. Install CRIU (brings in crit, protobuf, etc.)
sudo apt-get install -y criu

# 3. Verify
criu --version                     # should print 4.x (4.2.1 from the PPA)
which crit                         # /usr/bin/crit  (CRIU image tool)
ls /usr/lib/criu/cuda_plugin.so    # shipped by the PPA package
sudo criu check                    # "Looks good."
```

On a **non-privileged** node (only `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`,
no `CAP_SYS_ADMIN`), plain `criu check` aborts at kernel-feature detection
because it tries to create a throwaway network namespace.  Check such a node
with the flag that the `SEMIP_UNPRIVILEGED=1` paths use:

```bash
sudo criu check --unprivileged      # must get past kerndat
```

A residual complaint about a read-only `ns_last_pid` is expected from the
checker and does **not** block restore: `clone3(set_tid)` at the recorded PIDs
is authorized by `CAP_CHECKPOINT_RESTORE`.  See Complication 11 in
[`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).

### Also qualify the interpreter for `SEMIP_UNPRIVILEGED=1`

`criu check` says nothing about the *other* precondition of unprivileged
mode: the child zeroes its capabilities, losing `CAP_DAC_OVERRIDE`, so
from then on uid 0 can only read what `other` can.  Every directory on
the path to the interpreter needs `o+x` and its files `o+r`.  Check the
**base** prefix, since a venv shares its base interpreter's stdlib:

```bash
namei -l "$(python -c 'import sys; print(sys.base_prefix)')"
```

Any component without `o+x` (a `750` home is the usual culprit) fails the
precondition.  A system-wide Python (`/usr/lib/python3.12`,
`/usr/local/lib/python3.12/dist-packages`) satisfies it for free, which is
why pre-baked pod images never hit this.

The functional version of the same check — worth running once per node,
because it exercises the exact call that breaks — drops capabilities and
then does what vLLM's `in_the_same_node_as` does during `reinit_nccl`:

```bash
sudo python3 -c '
import ctypes
from multiprocessing import shared_memory, resource_tracker  # pre-import
libc = ctypes.CDLL("libc.so.6", use_errno=True)
for c in range(64): libc.prctl(24, c, 0, 0, 0)   # PR_CAPBSET_DROP
libc.prctl(47, 4, 0, 0, 0)                       # PR_CAP_AMBIENT_CLEAR_ALL
class H(ctypes.Structure): _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class D(ctypes.Structure): _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
libc.capset(ctypes.byref(H(0x20080522,0)), ctypes.byref((D*2)()))
s = shared_memory.SharedMemory(create=True, size=128); print("OK", s.name)
s.close(); s.unlink()'
```

Run it with the interpreter you will actually serve with.  `OK` means the
node is fit; a `BrokenPipeError` means the precondition is violated and a
TP>1 restore will **hang silently** in `reinit_nccl` (Complication 11).
Fix it before dumping: images are otherwise fine, but you will not find
out until a later restore.

The dump also needs the empty plugin directory it passes to `--libdir`,
which no CRIU package creates.  `_worker_criu_save` now creates it (via
`sudo` when the worker is not root), so this is only needed if you want
it in place ahead of the first dump:

```bash
sudo mkdir -p /usr/lib/criu/empty
```

See *Complication 7* in [`CRIU_PLUMBING.md`](./CRIU_PLUMBING.md) for why
the directory exists at all.

### Alternative: build from source (when apt mirrors are unreachable)

On hosts where `archive.ubuntu.com` and/or `ppa.launchpadcontent.net`
are blocked or flaky, the PPA path above will fail with connection
timeouts or `404`s.  GitHub is usually still reachable, so the
fallback is to build CRIU 4.2 and its dependencies from source.

Required tooling already on the box: `gcc`, `make`, `autoconf`,
`automake`, `libtool`, `pkg-config`, `protoc` (libprotoc 3.21.x),
`git`, `python3`.

```bash
mkdir -p /tmp/criu-build && cd /tmp/criu-build

# A. Fix-ups for headers/symlinks the system protobuf/libuuid packages
#    omit (only run the ones whose targets are actually missing).
sudo ln -sf /usr/lib/x86_64-linux-gnu/libprotoc.so.32 \
            /usr/lib/x86_64-linux-gnu/libprotoc.so
sudo ln -sf /usr/lib/x86_64-linux-gnu/libuuid.so.1 \
            /usr/lib/x86_64-linux-gnu/libuuid.so

# protobuf compiler headers (needed by protobuf-c) — match installed protoc:
git clone --depth 1 --branch v3.21.12 \
    https://github.com/protocolbuffers/protobuf.git protobuf-src
sudo cp -r protobuf-src/src/google/protobuf/compiler \
           /usr/include/google/protobuf/compiler

# uuid.h header (needed by CRIU)
git clone --depth 1 --branch v2.40.4 \
    https://github.com/util-linux/util-linux.git
sudo mkdir -p /usr/include/uuid
sudo cp util-linux/libuuid/src/uuid.h /usr/include/uuid/uuid.h

# B. Dependencies
git clone --depth 1 --branch libcap-2.73 \
    https://git.kernel.org/pub/scm/libs/libcap/libcap.git
( cd libcap && make -j$(nproc) && sudo make install prefix=/usr )

git clone --depth 1 --branch v1.5.0 \
    https://github.com/protobuf-c/protobuf-c.git
( cd protobuf-c && ./autogen.sh && ./configure --prefix=/usr \
    && make -j$(nproc) && sudo make install )

git clone --depth 1 --branch v1.3 https://github.com/libnet/libnet.git
( cd libnet && ./autogen.sh && ./configure --prefix=/usr \
    && make -j$(nproc) && sudo make install )

# C. CRIU 4.2 itself (also builds cuda_plugin.so)
git clone --depth 1 --branch v4.2 \
    https://github.com/checkpoint-restore/criu.git
cd criu
PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH" \
    make -j$(nproc)
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-criu PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-lib  PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-crit PREFIX=/usr

# D. Plugin dir (same as PPA path)
sudo mkdir -p /usr/lib/criu/empty
```

Verify:

```bash
criu --version          # Version: 4.2,  GitID: v4.2
which crit              # /usr/local/bin/crit  (note: not /usr/bin/crit)
ls -d /usr/lib/criu/empty
```

Notes:

- `crit` lands in `/usr/local/bin/` on the source build (vs `/usr/bin/`
  from the PPA), because it ships as a Python wheel installed by
  `install-crit`.
- The from-source path also produces `cuda_plugin.so` in the CRIU build
  tree — the CRIU CUDA infrastructure picks it up at dump time.  The PPA
  package already ships it as `/usr/lib/criu/cuda_plugin.so`, so the CUDA
  plugin is not on its own a reason to prefer the source build.
- More detailed build notes (and conditional fix-ups for systems missing
  even more headers) live in `instance_DESIGN.md` under
  *"CRIU Installation (v4.2, from source)"*.

---

## 2. Speculative draft models

`register.py` and `register_FCA.py` reference four speculative draft
models, all expected under `/data-fast/`:

- `/data-fast/spec-decode-qwen3-8b-search_r1`
- `/data-fast/spec-decode-qwen3-30b-search_r1`
- `/data-fast/qwen3-32b-bird-4096-3head`
- `/data-fast/qwen3-32b-longcontext-4096-3head`

Sync them from S3 (requires AWS credentials with read access to the
`ml-dev-sfc-or-dev-misc1-k8s` bucket):

```bash
aws s3 sync \
    s3://ml-dev-sfc-or-dev-misc1-k8s/snowflake_research/checkpoint/speculator/llama-3.3-70b/jaeseong/spec-decode-qwen3-8b-search_r1 \
    /data-fast/spec-decode-qwen3-8b-search_r1

aws s3 sync \
    s3://ml-dev-sfc-or-dev-misc1-k8s/snowflake_research/checkpoint/speculator/llama-3.3-70b/jaeseong/spec-decode-qwen3-30b-search_r1 \
    /data-fast/spec-decode-qwen3-30b-search_r1

aws s3 sync \
    s3://ml-dev-sfc-or-dev-misc1-k8s/snowflake_research/checkpoint/speculator/qwen3-32b-bird-4096-3head \
    /data-fast/qwen3-32b-bird-4096-3head

aws s3 sync \
    s3://ml-dev-sfc-or-dev-misc1-k8s/snowflake_research/checkpoint/speculator/qwen3-32b-longcontext-4096-3head \
    /data-fast/qwen3-32b-longcontext-4096-3head
```

`aws s3 sync` is idempotent — re-running it after a partial download
only fetches missing/changed objects, so it's safe to retry on flaky
connections.

Verify the destinations are non-empty:

```bash
ls /data-fast/spec-decode-qwen3-8b-search_r1
ls /data-fast/spec-decode-qwen3-30b-search_r1
ls /data-fast/qwen3-32b-bird-4096-3head
ls /data-fast/qwen3-32b-longcontext-4096-3head
```
