#!/usr/bin/env python3
import os, argparse, time, math, csv, socket
import torch
import deepspeed
import deepspeed.comm as dist

# ───────────────────────── helpers ──────────────────────────
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

# Reduce DS verbosity before import‑time logging fires
os.environ.setdefault("DS_VERBOSE", "1")
os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "error")

# ─────────────────────────  main  ───────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Distributed matrix‑multiply benchmark (CPU) using DeepSpeed"
    )
    # launcher bookkeeping
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank (injected by DeepSpeed launcher)")
    # matrix + timing parameters
    parser.add_argument("--size", type=int, default=1024, help="Matrix dimension (N×N)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--count", type=int, default=10, help="# timed iters")
    parser.add_argument("--warmup", type=int, default=2, help="# warm‑up iters")
    # communication/reduction switches
    parser.add_argument("--ccl", action="store_true",
                        help="Use dist.all_reduce (vs. inference_all_reduce)")
    parser.add_argument("--backend", choices=["mpi", "gloo", "ccl"],
                        default="mpi", help="Torch/DeepSpeed comm backend")
    # bookkeeping for CSV
    parser.add_argument("--outfile", default="results.csv", help="CSV file")
    parser.add_argument("--gpu_count", type=int, default=1)
    parser.add_argument("--accelerator", default="CPU")
    parser.add_argument("--launcher", default="impiexec")

    args = parser.parse_args()

    # map dtype
    dtype = {"fp32": torch.float32,
             "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]

    # ───── DeepSpeed init (backend now selectable) ─────
    # deepspeed.init_distributed(dist_backend=args.backend)
    deepspeed.init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    hostname, ip = socket.gethostname(), get_ip_address()

    device = torch.device("cpu")
    N = args.size

    # create / broadcast matrices
    if rank == 0:
        A = torch.rand(N, N, dtype=dtype, device=device)
        B = torch.rand(N, N, dtype=dtype, device=device)
    else:
        A = torch.empty(N, N, dtype=dtype, device=device)
        B = torch.empty(N, N, dtype=dtype, device=device)

    torch.distributed.broadcast(A, 0)
    torch.distributed.broadcast(B, 0)

    # partition inner dimension
    base, rem = divmod(N, world)
    k_start = rank * (base + 1) if rank < rem else rem * (base + 1) + (rank - rem) * base
    k_count = base + 1 if rank < rem else base
    k_end = k_start + k_count

    timings_mm, timings_ar = [], []
    total_iters = args.count + args.warmup
    final_C = None

    for i in range(total_iters):
        t0 = time.time()
        local_C = torch.matmul(A[:, k_start:k_end], B[k_start:k_end, :])
        t1 = time.time()

        final_C = local_C.clone()

        t2 = time.time()
        dist.all_reduce(final_C)       # OR inference_all_reduce via --ccl
        t3 = time.time()

        if i >= args.warmup:
            timings_mm.append(t1 - t0)
            timings_ar.append(t3 - t2)

    # stats
    def stats(v):
        avg = sum(v) / len(v)
        return min(v), max(v), avg, (
            math.sqrt(sum((x-avg)**2 for x in v)/len(v))/avg*100 if avg else 0
        )
    mm_min, mm_max, mm_avg, mm_sd = stats(timings_mm)
    ar_min, ar_max, ar_avg, ar_sd = stats(timings_ar)
    total = [m+a for m,a in zip(timings_mm, timings_ar)]
    tt_min, tt_max, tt_avg, tt_sd = stats(total)

    if rank == 0:
        # validation
        if not torch.allclose(final_C, torch.matmul(A, B), rtol=1e-3, atol=1e-5):
            print("❌ distributed result mismatch")
            return
        print("✅ result matches reference")
        print(f"Total:  min {tt_min:.6f}s  max {tt_max:.6f}s  avg {tt_avg:.6f}s  ±{tt_sd:.2f}%")
        print(f"Matmul: min {mm_min:.6f}s  max {mm_max:.6f}s  avg {mm_avg:.6f}s  ±{mm_sd:.2f}%")
        print(f"AllRed: min {ar_min:.6f}s  max {ar_max:.6f}s  avg {ar_avg:.6f}s  ±{ar_sd:.2f}%")

        header = ["Implementation","CPU_Count","Matrix_Size","Iterations",
                  "Matmul_t_min(s)","Matmul_t_max(s)","Matmul_t_avg(s)","Matmul_stddev(%)",
                  "Allreduce_t_min(s)","Allreduce_t_max(s)","Allreduce_t_avg(s)","Allreduce_stddev(%)",
                  "Total_t_min(s)","Total_t_max(s)","Total_t_avg(s)","Total_stddev(%)",
                  "GPU_Count","Accelerator","CCL","Launcher","Backend"]
        row = ["DeepSpeed", world, N, args.count,
               f"{mm_min:.6f}", f"{mm_max:.6f}", f"{mm_avg:.6f}", f"{mm_sd:.2f}",
               f"{ar_min:.6f}", f"{ar_max:.6f}", f"{ar_avg:.6f}", f"{ar_sd:.2f}",
               f"{tt_min:.6f}", f"{tt_max:.6f}", f"{tt_avg:.6f}", f"{tt_sd:.2f}",
               args.gpu_count, args.accelerator,
               "true" if args.ccl else "false", args.launcher, args.backend]
        outdir = os.path.dirname(args.outfile)
        if outdir:                       # '' when you pass just a filename
            os.makedirs(outdir, exist_ok=True)

        with open(args.outfile, "w", newline="") as f:
            csv.writer(f).writerows([header, row])
        with open(args.outfile, "w", newline="") as f:
            csv.writer(f).writerows([header, row])

if __name__ == "__main__":
    main()
