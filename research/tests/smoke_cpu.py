"""CPU smoke test: default-sized tri-plane field fitting a smooth colour function.

Run: python research/tests/smoke_cpu.py   (or via pytest: test_smoke)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from triplane_field import TriplaneField, TriplaneFieldConfig, make_field_optimizer

AABB = torch.tensor([[-1.5] * 3, [1.5] * 3])


def target_fn(x):
    r = torch.sin(2.0 * x[:, 0] + 1.0 * x[:, 1])
    g = torch.sin(3.0 * x[:, 1] - 1.0 * x[:, 2])
    b = torch.sin(2.5 * x[:, 2] + 0.5 * x[:, 0])
    return (0.5 * torch.stack([r, g, b], -1)).unsqueeze(1)  # [N,1,3]


def run(cfg, n=20000, iters=20, label=""):
    torch.manual_seed(0)
    means = (torch.rand(n, 3) * 2 - 1) * 1.3
    target = target_fn(means)
    sh0 = torch.nn.Parameter(torch.zeros(n, 1, 3))
    fld = TriplaneField(AABB, cfg)
    f_opt, f_sched = make_field_optimizer(fld, 1e-2, 1e-3, 1e-3, max_steps=iters)
    p_opt = torch.optim.Adam([sh0], lr=1e-2)
    losses, times = [], []
    for it in range(iters):
        t0 = time.perf_counter()
        fld.set_step(it)
        pred = sh0 + fld(means)["sh0"]
        loss = F.mse_loss(pred, target) + fld.regularization(1e-4, 0.0)
        assert torch.isfinite(loss), f"non-finite loss at iter {it}"
        f_opt.zero_grad()
        p_opt.zero_grad()
        loss.backward()
        f_opt.step()
        p_opt.step()
        f_sched.step()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    print(f"[{label}] loss start {losses[0]:.5f} end {losses[-1]:.5f} "
          f"time/iter {sum(times) / len(times):.3f}s")
    for p in list(fld.parameters()) + [sh0]:
        assert torch.isfinite(p).all(), "NaN/inf in parameters"
    return losses, times


def test_smoke():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    losses, _ = run(TriplaneFieldConfig(plane_cnn=True), label="C=32 R=128 cnn")
    assert losses[-1] < losses[0]
    run(
        TriplaneFieldConfig(resolutions=(64, 128, 256), fusion="concat"),
        iters=3,
        label="multires concat",
    )


if __name__ == "__main__":
    t = time.perf_counter()
    test_smoke()
    print(f"total {time.perf_counter() - t:.1f}s")
