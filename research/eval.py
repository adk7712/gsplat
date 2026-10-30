"""Evaluation protocol for the tri-plane study (NEW file).

Protocol (chosen to match the 3DGS paper's NeRF-Synthetic numbers; details in
research/README.md):
  * All 200 views of transforms_test.json at full 800x800, white background.
  * Renders clamped to [0, 1]; GT is the uint8 RGBA->white composite / 255.
  * PSNR: per-image -10*log10(MSE), then averaged over images (as 3DGS metrics.py).
  * SSIM: 11x11 Gaussian window, sigma 1.5, data_range 1 (torchmetrics; same
    formula as 3DGS utils/loss_utils.ssim), averaged over images.
  * LPIPS-VGG and LPIPS-Alex via torchmetrics with normalize=True, i.e. inputs in
    [0, 1] are mapped to [-1, 1] as the LPIPS authors intend. NOTE: the 3DGS repo's
    lpipsPyTorch feeds [0, 1] images without that rescaling, so its LPIPS values
    are not directly comparable; 3DGS reports only PSNR for NeRF-Synthetic anyway.
  * FPS: rasterisation-only, batch 1, 800x800, over all test views after 10
    warm-up renders, with torch.cuda.synchronize(). The field is evaluated ONCE
    before timing ("baked"), since means are static at test time; its one-off
    query cost is reported separately as field_query_ms.
"""

import os
import time
from typing import Callable, Dict, Sequence

import numpy as np
import torch


class ImageMetrics:
    """PSNR / SSIM / LPIPS-VGG / LPIPS-Alex on [0, 1] images of shape [1, 3, H, W]."""

    def __init__(self, device: torch.device):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(device)
        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        pred = pred.clamp(0.0, 1.0)
        mse = torch.mean((pred - gt) ** 2)
        return {
            "psnr": float(-10.0 * torch.log10(mse)),
            "ssim": float(self.ssim(pred, gt)),
            "lpips_vgg": float(self.lpips_vgg(pred, gt)),
            "lpips_alex": float(self.lpips_alex(pred, gt)),
        }


@torch.no_grad()
def evaluate(
    render_fn: Callable[[int], torch.Tensor],
    gt_images: torch.Tensor,
    device: torch.device,
    render_dir: str = "",
    save_indices: Sequence[int] = (0, 50, 100, 150),
    fps_warmup: int = 10,
) -> Dict[str, float]:
    """render_fn(i) -> [H, W, 3] float render of test view i (attributes pre-baked).
    gt_images: uint8 [N, H, W, 3] (CPU or GPU)."""
    n = gt_images.shape[0]
    metrics = ImageMetrics(device)
    per_image = []
    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
    for i in range(n):
        pred = render_fn(i).clamp(0, 1)
        gt = gt_images[i].to(device).float() / 255.0
        per_image.append(
            metrics(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        )
        if render_dir and i in save_indices:
            import imageio.v2 as imageio

            pair = torch.cat([pred, gt], dim=1)  # render | GT side by side
            imageio.imwrite(
                os.path.join(render_dir, f"test_{i:03d}.png"),
                (pair.cpu().numpy() * 255).round().astype(np.uint8),
            )
    out = {k: float(np.mean([m[k] for m in per_image])) for k in per_image[0]}

    # FPS: separate timed pass so metric computation does not pollute timing.
    use_cuda = device.type == "cuda"
    for i in range(min(fps_warmup, n)):
        render_fn(i)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        render_fn(i)
    if use_cuda:
        torch.cuda.synchronize()
    out["fps"] = n / (time.perf_counter() - t0)
    out["num_test_views"] = n
    return out
