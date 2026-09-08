#!/usr/bin/env python3
"""Check a CRIU image's recorded task ids against the ones live on this node.

    sudo python3 scripts/pidcheck.py /data-fast/image-cache/qwen_27b/image
    python3 scripts/pidcheck.py <image> --burn        # make room, then restore
    python3 scripts/pidcheck.py --burn-to 200000      # dump side, before init

Use this before a restore in reduced-capability mode (``SEMIP_UNPRIVILEGED=1``),
which has no private PID namespace: CRIU recreates every task at its recorded
id with ``clone3(set_tid)``, so every one of them must be free on the host.

A PID is only the id of a thread group's leader, and leader ids and thread ids
are allocated from a single per-namespace counter.  So a TP2 image needs its
three leaders *and* the ~900 thread ids they recorded, and an unrelated
200-thread process sitting anywhere in that range blocks the restore just as a
same-PID process would.  ``ps`` and a plain ``/proc`` scan both hide this,
because threads appear only under ``/proc/<pid>/task``.

CRIU reports the two cases through different code paths and names neither the
id nor the occupant:

    Can't fork for 47619: File exists                 # a leader
    pie: 2103: Unable to create a thread: -17         # any other thread

``--burn`` advances the namespace's PID counter past the image's highest
recorded id by forking throwaway children -- it consumes fresh numbers, it does
not free occupied ones, and it cannot move a process that is already running.
So it fixes "my launcher landed in the image's range" and nothing else; ids
held by a live process have to be freed by that process exiting.  The counter
is namespace-global, so burning in this script carries over to the restore
launched after it.  ``/proc/sys/kernel/ns_last_pid`` would do the same in one
write, but it is read-only on the nodes this mode exists for.

The durable fix is dump-side: ``--burn-to N`` before the cold start, so the
image records ids far above anything a container hands out in normal operation.
That form needs no image and no root -- it only advances the counter.

Like ``imgdiff.py`` this imports nothing from the package and needs no GPU --
just ``crit`` (from the CRIU install) and root to read the image.
"""

import json
import os
import subprocess
import sys

BURN_MARGIN = 2000  # headroom for the restore driver and its threads


def decode_pstree(path):
    cmd = ["crit", "decode", "-i", path]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        sys.exit("crit decode failed (rc=%d); root is needed to read the image:"
                 "\n%s" % (res.returncode, res.stderr.decode("utf-8", "replace")))
    return json.loads(res.stdout.decode("utf-8", "replace"))


def recorded_tasks(doc):
    """(leaders, tids) recorded in pstree.img."""
    leaders, tids = [], set()
    for ent in doc.get("entries", []):
        if ent.get("pid") is None:
            continue
        leaders.append(int(ent["pid"]))
        tids.add(int(ent["pid"]))
        tids.update(int(t) for t in (ent.get("threads") or []))
    return leaders, tids


def live_task_ids():
    """{task id: leader pid} for every task live in this PID namespace."""
    live = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            for tid in os.listdir("/proc/%s/task" % entry):
                live[int(tid)] = int(entry)
        except OSError:
            continue                     # exited while we walked it
    return live


def proc_field(pid, name):
    try:
        with open("/proc/%d/%s" % (pid, name)) as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def ranges(ids):
    out = []
    for v in sorted(ids):
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return ", ".join("%d-%d" % (a, b) if a != b else str(a) for a, b in out)


def read_sysctl(name):
    try:
        with open("/proc/sys/kernel/%s" % name) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def read_counter():
    return read_sysctl("ns_last_pid")


def burn_past(target):
    """Fork throwaway children until the counter passes ``target``."""
    pid_max = read_sysctl("pid_max") or 32768
    if target >= pid_max:
        sys.exit("target %d exceeds pid_max %d, so no range on this node is "
                 "safe from collisions" % (target, pid_max))
    last = read_counter()
    print("counter before : %s" % last)
    burned = 0
    while True:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        burned += 1
        if pid >= target:
            break
        if pid < (last or 0) and burned > pid_max:
            sys.exit("counter wrapped without reaching %d" % target)
    print("counter after  : %d  (%d id(s) burned)" % (pid, burned))
    print("\nStart the next process now -- anything spawned after this lands "
          "above %d.  Already-running processes did NOT move, so an occupant "
          "reported above still has to exit." % target)


USAGE = ("usage: pidcheck.py <image-dir-or-model-dir> [--burn]\n"
         "       pidcheck.py --burn-to <id>")


def parse_argv(argv):
    """(positional, burn, burn_to) -- rejects anything else."""
    positional, burn, burn_to = [], False, None
    rest = list(argv)
    while rest:
        arg = rest.pop(0)
        if arg == "--burn":
            burn = True
        elif arg == "--burn-to" or arg.startswith("--burn-to="):
            value = (arg.split("=", 1)[1] if "=" in arg
                     else (rest.pop(0) if rest else ""))
            if not value.isdigit():
                sys.exit("--burn-to needs a task id, e.g. --burn-to 200000")
            burn_to = int(value)
        elif arg.startswith("-"):
            sys.exit(USAGE)
        else:
            positional.append(arg)
    return positional, burn, burn_to


def main():
    args, burn, burn_to = parse_argv(sys.argv[1:])

    # Dump-side form: no image exists yet, so there is nothing to compare
    # against -- just move the counter so the tree about to be created records
    # ids nothing will squat on later.
    if burn_to is not None and not args:
        print("BURN  (target %d)" % burn_to)
        burn_past(burn_to)
        sys.exit(0)
    if len(args) != 1:
        sys.exit(USAGE)

    target = args[0]
    if target.endswith(".img"):
        pstree = target
    else:
        pstree = os.path.join(target, "pstree.img")
        if not os.path.exists(pstree):
            pstree = os.path.join(target, "image", "pstree.img")
    if not os.path.exists(pstree):
        sys.exit("no pstree.img under %s" % target)

    leaders, tids = recorded_tasks(decode_pstree(pstree))
    if not tids:
        sys.exit("pstree.img recorded no tasks (unexpected image layout)")

    print("=" * 78)
    print("CRIU image: %s" % pstree)
    print("=" * 78)
    print("thread group leaders : %s" % leaders)
    print("task ids required    : %d  (range %d-%d)"
          % (len(tids), min(tids), max(tids)))
    print("recorded id ranges   : %s" % ranges(tids)[:500])
    print("ns_last_pid now      : %s   (pid_max %s)"
          % (read_counter(), read_sysctl("pid_max")))
    print()

    live = live_task_ids()
    taken_by = {}
    for tid in sorted(tids.intersection(live)):
        taken_by.setdefault(live[tid], []).append(tid)

    if taken_by:
        print("-" * 78)
        print("OCCUPIED  (%d id(s) across %d process(es))"
              % (sum(len(v) for v in taken_by.values()), len(taken_by)))
        print("-" * 78)
        for owner, taken in sorted(taken_by.items()):
            state = proc_field(owner, "stat").rsplit(")", 1)[-1].split()
            print("  pid %-7d %-20s state=%-2s threads=%-4s" %
                  (owner, proc_field(owner, "comm"),
                   state[0] if state else "?",
                   len(os.listdir("/proc/%d/task" % owner))
                   if os.path.isdir("/proc/%d/task" % owner) else "?"))
            print("      takes %d recorded id(s): %s"
                  % (len(taken), ranges(taken)[:200]))
        print()
        print("A restore will fail with EEXIST on the first of these.")
        print("Zombies clear themselves once reaped; a live process must exit,")
        print("or the image must be re-dumped with its ids placed higher.")
    else:
        print("No collisions: every recorded task id is free on this node.")

    if burn or burn_to is not None:
        goal = burn_to if burn_to is not None else max(tids) + BURN_MARGIN
        print()
        print("-" * 78)
        print("BURN  (target %d%s)"
              % (goal, "" if burn_to is not None
                 else " = max recorded id + %d" % BURN_MARGIN))
        print("-" * 78)
        burn_past(goal)

    sys.exit(1 if taken_by else 0)


if __name__ == "__main__":
    main()
