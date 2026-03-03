"""End-to-end test for NCCLEngine bucket-packed broadcast."""
import os, sys, time, socket
import torch, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

MASTER_ADDR = "127.0.0.1"
NUM_WEIGHTS = 20
WEIGHT_SHAPES = [(512, 256)] * NUM_WEIGHTS
WEIGHT_DTYPE = torch.bfloat16
BUCKET_SIZE = 4 * 1024 * 1024  # 4MB — small for fast test


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sender_fn(port, ref_tensors, world_size):
    from arctic_inference.server.weight_sync import NCCLEngine

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    engine = NCCLEngine(
        master_addr=MASTER_ADDR, master_port=port,
        rank=0, world_size=world_size, device=device,
        bucket_size=BUCKET_SIZE,
    )

    weights = (
        ("w%d" % j, ref_tensors[j].to(device))
        for j in range(NUM_WEIGHTS)
    )

    result = engine.send_weights(weights)
    print("  Sender: sent %d in %.3fs (%d buckets)" % (
        result["params_sent"], result["elapsed"], result["buckets"]))
    engine.destroy()


def _receiver_fn(rank, port, ref_tensors, world_size, results_dict):
    from arctic_inference.server.weight_sync import NCCLEngine

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    engine = NCCLEngine(
        master_addr=MASTER_ADDR, master_port=port,
        rank=rank, world_size=world_size, device=device,
        bucket_size=BUCKET_SIZE,
    )

    t0 = time.time()
    received = {}
    for name, tensor in engine.receive_weights():
        received[name] = tensor.clone()
    elapsed = time.time() - t0

    ok = True
    for j in range(NUM_WEIGHTS):
        key = "w%d" % j
        if key not in received:
            print("  FAIL recv=%d missing %s" % (rank, key))
            ok = False
            continue
        if not torch.equal(ref_tensors[j], received[key].cpu()):
            print("  FAIL recv=%d %s mismatch" % (rank, key))
            ok = False

    results_dict[rank] = {"ok": ok, "elapsed": elapsed}
    engine.destroy()


def _worker_fn(rank, n_receivers, port, ref_tensors, results_dict):
    world_size = 1 + n_receivers
    if rank == 0:
        _sender_fn(port, ref_tensors, world_size)
    else:
        _receiver_fn(rank, port, ref_tensors, world_size, results_dict)


def main():
    n_receivers = int(os.environ.get("NUM_RECEIVERS", "2"))
    total = 1 + n_receivers
    n_gpus = torch.cuda.device_count()
    if n_gpus < total:
        print("Need %d GPUs, have %d" % (total, n_gpus))
        sys.exit(1)
    print("NCCLEngine broadcast test: 1 sender, %d receivers, %d weights" % (
        n_receivers, NUM_WEIGHTS))
    port = find_free_port()
    ref_tensors = [torch.randn(s, dtype=WEIGHT_DTYPE) for s in WEIGHT_SHAPES]
    manager = mp.Manager()
    results_dict = manager.dict()
    mp.spawn(
        _worker_fn,
        args=(n_receivers, port, ref_tensors, results_dict),
        nprocs=total,
        join=True,
    )
    all_ok = True
    for r in range(1, total):
        res = results_dict.get(r, {})
        ok = res.get("ok", False)
        print("  Receiver %d: %s  elapsed=%.3fs" % (
            r, "PASS" if ok else "FAIL", res.get("elapsed", -1)))
        if not ok:
            all_ok = False
    total_bytes = sum(s[0] * s[1] * 2 for s in WEIGHT_SHAPES)
    avg_elapsed = sum(results_dict[r]["elapsed"] for r in range(1, total)) / n_receivers
    bw = total_bytes / avg_elapsed / 1e9 if avg_elapsed > 0 else 0
    print("Total: %.1fMB  Avg=%.3fs  BW=%.2fGB/s" % (
        total_bytes / 1e6, avg_elapsed, bw))
    print("RESULT: %s" % ("PASSED" if all_ok else "FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
