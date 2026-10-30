"""GPU micro-benchmark of the tri-plane field's per-step cost (NEW file).

Written to explain the ~2x training slowdown seen at Checkpoint B (+34 ms/step at
~313k Gaussians, growing with N). Times one training step of the FIELD ONLY
(effective planes -> query -> loss incl. TV -> backward -> Adam step), for both
plane samplers, with/without the plane CNN, for several N and two spatial
distributions of means. Also prints the top CUDA kernels for the slowest case.

  python research/bench_field.py --result_dir results/bench
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triplane_field import TriplaneField, TriplaneFieldConfig, make_field_optimizer  # noqa: E402


def make_means(n: int, kind: str, device) -> torch.Tensor:
    if kind == "uniform":
        return (torch.rand(n, 3, device=device) * 2 - 1) * 1.3
    # "clustered": points on the faces of a few boxes (object-like surfaces), which is
    # how trained Gaussians are distributed; many Gaussians share plane texels.
    g = torch.Generator(device="cpu").manual_seed(0)
    centers = (torch.rand(8, 3, generator=g) * 2 - 1) * 0.8
    sizes = torch.rand(8, 3, generator=g) * 0.5 + 0.1
    box = torch.randint(0, 8, (n,), generator=g)
    p = (torch.rand(n, 3, generator=g) * 2 - 1)
    face = torch.randint(0, 3, (n,), generator=g)
    p[torch.arange(n), face] = torch.sign(p[torch.arange(n), face])  # snap to a face
    pts = centers[box] + p * sizes[box] + torch.randn(n, 3, generator=g) * 0.002
    return pts.to(device)


def bench(sampler: str, cnn: bool, means: torch.Tensor, steps: int, profile: bool = False):
    dev = means.device
    fld = TriplaneField(
        torch.tensor([[-1.5] * 3, [1.5] * 3]),
        TriplaneFieldConfig(plane_cnn=cnn, sampler=sampler),
    ).to(dev)
    torch.nn.init.normal_(fld.mlp[-1].weight, std=0.01)  # as after a few steps
    opt, sched = make_field_optimizer(fld, 1.6e-3, 1.6e-4, 1.6e-4, 30_000)
    target = torch.rand(len(means), 1, 3, device=dev)

    def step():
        planes = fld.effective_planes()
        out = fld(means, planes)["sh0"]
        loss = (out - target).abs().mean() + fld.regularization(1e-4, 0.0, planes)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / steps
    table = None
    if profile:
        from torch.profiler import ProfilerActivity, profile as tprof

        with tprof(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                step()
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=8)
    return ms, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="results/bench")
    ap.add_argument("--n", type=int, nargs="+", default=[50_000, 150_000, 313_000])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(dev), "| torch", torch.__version__, flush=True)
    rows = []
    for kind in ("clustered", "uniform"):
        for n in args.n:
            means = make_means(n, kind, dev)
            for cnn in (False, True):
                for sampler in ("grid_sample", "embedding"):
                    ms, _ = bench(sampler, cnn, means, args.steps)
                    rows.append(dict(dist=kind, n=n, cnn=cnn, sampler=sampler, ms_per_step=ms))
                    print(f"{kind:9s} N={n:7d} cnn={int(cnn)} {sampler:11s} {ms:7.2f} ms/step", flush=True)
    for sampler in ("grid_sample", "embedding"):
        _, table = bench(sampler, False, make_means(max(args.n), "clustered", dev), 5, profile=True)
        print(f"\nTop CUDA kernels, {sampler}, clustered N={max(args.n)}:\n{table}", flush=True)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "bench_field.json"), "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
