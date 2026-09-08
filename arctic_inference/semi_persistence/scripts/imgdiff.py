#!/usr/bin/env python3
"""Check a CRIU image's file-backed mappings against the local filesystem.

    sudo python3 scripts/imgdiff.py /data-fast/image-cache/qwen_27b/image

Use this before attempting a cross-node restore.  CRIU records every
file-backed mapping by absolute path *and* size and re-validates the size
when it reopens the file (criu/files-reg.c: "File %s has bad size %"), so
one mapped file of a different length aborts the whole restore.  There is
no up-front check for this the way there is for ``vllm_config`` and
``model_dir``, and CRIU stops at the first bad mapping rather than
reporting them all -- hence this script.  See *Cross-node restore also
needs a byte-identical environment* in ``skills/semi-p_DESIGN.md``.

CRIU compares only the size, so this also compares the ELF build-ID CRIU
recorded: a same-size but different-build library would pass CRIU's check
and then map the wrong text pages into the restored process.

Unlike everything else in ``scripts/``, this needs no GPU, no vLLM and no
``sys.path`` bootstrap -- it imports nothing from the package.  It does
need ``crit`` (from the CRIU install) and root to read the image.

Observed files.img JSON shape (crit 4.2.1):
  { "magic": "FILES",
    "entries": [
      { "type": "REG", "id": 1,
        "reg": { "id": 1, "flags": "", "pos": 0, "fown": {...},
                 "name": "/abs/path", "size": 30598912, "mode": 33261,
                 "build_id": [4064338300, 3061931570, ...] } },
      { "type": "MEMFD", "id": 2, "memfd": {...} },
      ...
    ] }
The payload lives under a key equal to lowercase(type).  "size" is optional
(absent for /dev/null, append-mode logs and directories).  "build_id" is
optional and is an array of little-endian uint32 words.

Files that were unlinked at dump time are carried inside the image itself as
"ghost" files (remap-fpath.img -> remap_type GHOST, contents in
ghost-file-*.img).  CRIU recreates those at restore, so they are reported
separately and never counted as something to sync.
"""

import json
import os
import stat
import struct
import subprocess
import sys
from collections import defaultdict

S_IFMT = 0o170000
S_IFREG = 0o100000
PT_NOTE = 4
NT_GNU_BUILD_ID = 3


def decode_files_img(path):
    cmd = ["crit", "decode", "-i", path]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        sys.exit("crit decode failed (rc=%d):\n%s"
                 % (res.returncode, res.stderr.decode("utf-8", "replace")))
    return json.loads(res.stdout.decode("utf-8", "replace"))


def ghost_ids(img_path):
    """Ids remapped to a GHOST file: CRIU recreates these, do not sync them."""
    remap = os.path.join(os.path.dirname(img_path), "remap-fpath.img")
    if not os.path.exists(remap):
        return set()
    cmd = ["crit", "decode", "-i", remap]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        return set()
    doc = json.loads(res.stdout.decode("utf-8", "replace"))
    return {e["orig_id"] for e in doc.get("entries", [])
            if e.get("remap_type") == "GHOST" and "orig_id" in e}


def reg_entries(doc):
    """Yield the 'reg' payload of every REG entry carrying a name and a size."""
    for ent in doc.get("entries", []):
        etype = ent.get("type")
        if etype != "REG":
            continue
        payload = ent.get(etype.lower())
        if not isinstance(payload, dict):
            continue
        if "name" not in payload or "size" not in payload:
            continue
        # Only real regular files are size-checked by CRIU.
        mode = payload.get("mode")
        if mode is not None and (mode & S_IFMT) != S_IFREG:
            continue
        yield payload


def abspath(name):
    """CRIU stores names rooted at the mount; normalise to an absolute path."""
    if not name.startswith("/"):
        name = "/" + name
    return os.path.normpath(name)


def image_build_id(reg):
    words = reg.get("build_id")
    if not words:
        return None
    return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words).hex()


def local_build_id(path):
    """Read NT_GNU_BUILD_ID out of the local ELF, or None if unavailable."""
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(64)
            if len(hdr) < 64 or hdr[:4] != b"\x7fELF":
                return None
            is64 = hdr[4] == 2
            little = hdr[5] == 1
            end = "<" if little else ">"
            if is64:
                phoff = struct.unpack_from(end + "Q", hdr, 0x20)[0]
                phentsize = struct.unpack_from(end + "H", hdr, 0x36)[0]
                phnum = struct.unpack_from(end + "H", hdr, 0x38)[0]
            else:
                phoff = struct.unpack_from(end + "I", hdr, 0x1C)[0]
                phentsize = struct.unpack_from(end + "H", hdr, 0x2A)[0]
                phnum = struct.unpack_from(end + "H", hdr, 0x2C)[0]
            if not phoff or not phnum:
                return None
            fh.seek(phoff)
            phdrs = fh.read(phentsize * phnum)
            for i in range(phnum):
                ph = phdrs[i * phentsize:(i + 1) * phentsize]
                if len(ph) < phentsize:
                    break
                ptype = struct.unpack_from(end + "I", ph, 0)[0]
                if ptype != PT_NOTE:
                    continue
                if is64:
                    off = struct.unpack_from(end + "Q", ph, 0x08)[0]
                    fsz = struct.unpack_from(end + "Q", ph, 0x20)[0]
                else:
                    off = struct.unpack_from(end + "I", ph, 0x04)[0]
                    fsz = struct.unpack_from(end + "I", ph, 0x10)[0]
                fh.seek(off)
                notes = fh.read(fsz)
                pos = 0
                while pos + 12 <= len(notes):
                    namesz, descsz, ntype = struct.unpack_from(end + "III", notes, pos)
                    pos += 12
                    name = notes[pos:pos + namesz]
                    pos += (namesz + 3) & ~3
                    desc = notes[pos:pos + descsz]
                    pos += (descsz + 3) & ~3
                    if ntype == NT_GNU_BUILD_ID and name.rstrip(b"\x00") == b"GNU":
                        return desc.hex()
    except OSError:
        return None
    return None


def classify(path, expected):
    try:
        st = os.stat(path)           # follow symlinks: CRIU opens the target
    except FileNotFoundError:
        return "ABSENT", None, None
    except NotADirectoryError:
        return "ABSENT", None, None
    except OSError as exc:
        return "UNREADABLE", None, str(exc)
    if not stat.S_ISREG(st.st_mode):
        return "NOT-A-REGULAR-FILE", st.st_size, None
    if st.st_size != expected:
        return "SIZE MISMATCH", st.st_size, None
    return "OK", st.st_size, None


def tree_roots(paths, depth=2):
    roots = set()
    for p in paths:
        parts = [c for c in p.split("/") if c]
        roots.add("/" + "/".join(parts[:depth]) if parts else "/")
    return sorted(roots)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: imgdiff.py <image-dir-or-files.img>")
    target = sys.argv[1]
    img = target if target.endswith(".img") else os.path.join(target, "files.img")

    print("=" * 78)
    print("CRIU image: %s" % img)
    print("=" * 78)

    doc = decode_files_img(img)
    ghosts = ghost_ids(img)
    type_counts = defaultdict(int)
    for ent in doc.get("entries", []):
        type_counts[ent.get("type")] += 1
    print("total entries in files.img : %d" % len(doc.get("entries", [])))
    print("entry types                : %s"
          % ", ".join("%s=%d" % (k, v) for k, v in sorted(type_counts.items())))

    results = []
    seen = set()
    bid_mismatch = []
    bid_checked = 0
    ghost_rows = []
    for reg in reg_entries(doc):
        path = abspath(reg["name"])
        expected = reg["size"]
        if (path, expected) in seen:
            continue
        seen.add((path, expected))
        if reg.get("id") in ghosts:
            ghost_rows.append((path, expected))
            continue
        status, local, err = classify(path, expected)
        results.append((status, path, expected, local, err))
        # Content check for files that passed the size check.
        want_bid = image_build_id(reg)
        if status == "OK" and want_bid:
            got_bid = local_build_id(path)
            if got_bid is not None:
                bid_checked += 1
                if got_bid != want_bid:
                    bid_mismatch.append((path, want_bid, got_bid))

    checked = len(results)
    print("regular-file mappings with a recorded size (deduped): %d" % checked)
    print("of those, ELF build-IDs verifiable and compared      : %d" % bid_checked)
    print()

    if ghost_rows:
        print("-" * 78)
        print("GHOST files carried inside the image (CRIU recreates, DO NOT sync)  (%d)"
              % len(ghost_rows))
        print("-" * 78)
        for path, expected in sorted(ghost_rows):
            print("  %s  (%d bytes, contents stored in ghost-file-*.img)" % (path, expected))
        print()

    buckets = defaultdict(list)
    for r in results:
        buckets[r[0]].append(r)

    bad_paths = []
    for status in ("SIZE MISMATCH", "ABSENT", "NOT-A-REGULAR-FILE", "UNREADABLE"):
        rows = sorted(buckets.get(status, []), key=lambda r: r[1])
        if not rows:
            continue
        print("-" * 78)
        print("%s  (%d)" % (status, len(rows)))
        print("-" * 78)
        for _, path, expected, local, err in rows:
            bad_paths.append(path)
            if status == "ABSENT":
                print("  %s\n      image expects %d bytes, local: MISSING" % (path, expected))
            elif status == "UNREADABLE":
                print("  %s\n      image expects %d bytes, local: %s" % (path, expected, err))
            else:
                print("  %s\n      image expects %d bytes, local %d bytes (delta %+d)"
                      % (path, expected, local, local - expected))
        print()

    if bid_mismatch:
        print("-" * 78)
        print("BUILD-ID MISMATCH (size matches, contents differ)  (%d)" % len(bid_mismatch))
        print("-" * 78)
        for path, want, got in sorted(bid_mismatch):
            bad_paths.append(path)
            print("  %s\n      image build-id %s\n      local build-id %s" % (path, want, got))
        print()

    print("=" * 78)
    print("SUMMARY for %s" % img)
    print("=" * 78)
    print("  OK                 : %d" % (len(buckets.get("OK", [])) - len(bid_mismatch)))
    print("  SIZE MISMATCH      : %d" % len(buckets.get("SIZE MISMATCH", [])))
    print("  ABSENT             : %d" % len(buckets.get("ABSENT", [])))
    print("  BUILD-ID MISMATCH  : %d" % len(bid_mismatch))
    print("  NOT-A-REGULAR-FILE : %d" % len(buckets.get("NOT-A-REGULAR-FILE", [])))
    print("  UNREADABLE         : %d" % len(buckets.get("UNREADABLE", [])))
    print("  GHOST (in-image)   : %d  [not a problem]" % len(ghost_rows))
    print("  ------------------------")
    print("  TOTAL CHECKED      : %d" % checked)
    print("  TOTAL PROBLEMS     : %d" % len(bad_paths))
    print()

    if bad_paths:
        print("Distinct top-level trees needing sync from the dump node:")
        for root in tree_roots(bad_paths, depth=2):
            n = sum(1 for p in bad_paths if p.startswith(root + "/"))
            print("  %-50s (%d file(s))" % (root, n))
        print()
        print("Distinct parent directories of affected files:")
        for d in sorted({os.path.dirname(p) for p in bad_paths}):
            n = sum(1 for p in bad_paths if os.path.dirname(p) == d)
            print("  %-70s (%d)" % (d, n))
        print()
    else:
        print("No problems found: every recorded mapping matches this node.")


if __name__ == "__main__":
    main()
