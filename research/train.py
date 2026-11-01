"""Compact NeRF-Synthetic trainer for the tri-plane study (NEW file).

Why not examples/simple_trainer.py? It only parses COLMAP/NCore data and carries
many unrelated features (compression, bilateral grid, viewer, distributed). This
trainer reuses gsplat's public pieces unchanged (rasterization, DefaultStrategy /
MCMCStrategy, losses) and adds exactly two things:
  1. a Blender/NeRF-Synthetic protocol matching the 3DGS paper (research/blender.py);
  2. the optional tri-plane attribute field (research/triplane_field.py).
With ``field=false`` it is plain gsplat ("vanilla").

Usage:
  python research/train.py --config research/configs/vanilla.json \
      --data_dir data/nerf_synthetic/lego --scene lego --seed 0 --result_dir results/local
"""

import argparse
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field as dc_field  # `field` is also a Config key
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender import load_split, scene_extent  # noqa: E402
from triplane_field import (  # noqa: E402
    TriplaneField,
    TriplaneFieldConfig,
    make_field_optimizer,
)

SH_C0 = 0.28209479177387814


@dataclass
class Config:
    variant: str = "vanilla"
    data_dir: str = "data/nerf_synthetic/lego"
    scene: str = "lego"
    seed: int = 0
    result_dir: str = "results/local"
    iters: int = 30_000
    # --- 3DGS NeRF-Synthetic protocol -------------------------------------------
    white_bg: bool = True
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_num_pts: int = 100_000  # 3DGS: random init, no SfM points for Blender
    init_extent: float = 1.3  # points uniform in [-1.3, 1.3]^3 (3DGS)
    init_opacity: float = 0.1
    init_scale: float = 1.0  # multiplier on the kNN-based initial scale
    ssim_lambda: float = 0.2
    # --- gsplat defaults (examples/simple_trainer.py) ---------------------------
    strategy: str = "default"  # "default" | "mcmc"
    mcmc_cap_max: int = 300_000
    opacity_reg: float = 0.0  # mcmc preset uses 0.01
    scale_reg: float = 0.0  # mcmc preset uses 0.01
    absgrad: bool = False
    antialiased: bool = False
    packed: bool = False
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    # --- tri-plane field (all NEW) ------------------------------------------------
    field: bool = False
    field_channels: int = 32
    field_res: int = 128
    field_multires: bool = False  # True -> resolutions {64, 128, 256}
    field_fusion: str = "product"  # "product" | "concat"
    field_mlp_width: int = 64
    field_opacity: bool = False  # also predict an opacity-logit residual
    # Field queries use detached means by default, so the colour field cannot pull
    # Gaussians around: geometry is optimised exactly as in vanilla and the
    # comparison isolates the appearance model. Flip to let gradients reach means.
    field_grad_to_means: bool = False
    # Opacity gate (added after Phase 1): scale each Gaussian's colour residual by its own
    # (detached) opacity, so faint, low-opacity Gaussians (suspected "ghost" floaters that
    # the CNN-smoothed planes tinted, see research/analysis/cnn_fluctuation_report.md) get
    # little field colour while solid Gaussians get almost all of it. Detached so the field
    # cannot push opacities up or down to change its own gate.
    field_opacity_gate: bool = False
    field_aabb: List[float] = dc_field(
        default_factory=lambda: [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
    )
    plane_cnn: bool = False
    plane_cnn_off_after: int = -1
    tv_weight: float = 1e-4
    l1_weight: float = 0.0
    reg_on_raw: bool = False  # ablation: regularise raw planes (old 4D behaviour)
    plane_lr: float = 1.6e-3  # 4DGaussians grid_lr_init
    cnn_lr: float = 1.6e-4  # 4DGaussians deformation_lr_init
    mlp_lr: float = 1.6e-4
    # --- outputs ------------------------------------------------------------------
    save_render_indices: List[int] = dc_field(default_factory=lambda: [0, 50, 100, 150])
    # Diagnostics (added after Phase 1): save final per-Gaussian means, opacities, stored sh0
    # and field sh0 residual as float16 .npz under <result_dir>/gaussians/.
    save_gaussians: bool = False
    # Sparse-view training (added after Phase 1): use only this many training views, picked
    # evenly spaced over transforms_train.json (see select_views). -1 = all 100 views. The
    # test set is unchanged (all 200 views). Scene scale is computed from the views used.
    train_views: int = -1
    log_every: int = 1000


def parse_config() -> Config:
    """Defaults <- --config JSON <- explicit CLI flags."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    defaults = {}
    if known.config:
        with open(known.config) as f:
            defaults = json.load(f)
        unknown = set(defaults) - {f.name for f in dataclasses.fields(Config)}
        if unknown:
            raise ValueError(f"unknown keys in {known.config}: {sorted(unknown)}")

    ap = argparse.ArgumentParser(parents=[pre])
    for f in dataclasses.fields(Config):
        default = defaults.get(
            f.name,
            f.default if f.default is not dataclasses.MISSING else f.default_factory(),
        )
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        elif f.type in (List[int], List[float]):
            elem = int if f.type == List[int] else float
            ap.add_argument(f"--{f.name}", type=elem, nargs="+", default=default)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=default)
    args = vars(ap.parse_args())
    args.pop("config")
    return Config(**args)


def select_views(n_total: int, k: int) -> List[int]:
    """Evenly spaced view indices, e.g. 12 of 100 -> [0, 9, 18, ..., 99]. Deterministic and
    independent of the seed, so every variant and seed trains on the same views."""
    if k < 0 or k >= n_total:
        return list(range(n_total))
    return sorted({int(round(i)) for i in np.linspace(0, n_total - 1, k)})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def knn_mean_dist(x: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """sqrt(mean squared distance to k nearest neighbours), as 3DGS's distCUDA2."""
    out = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i : i + chunk], x)  # [c, N]
        d2 = d.topk(k + 1, largest=False).values[:, 1:] ** 2  # drop self
        out.append(d2.mean(-1).clamp_min(1e-7).sqrt())
    return torch.cat(out)


def create_splats(cfg: Config, scene_scale: float, device):
    N = cfg.init_num_pts
    means = (torch.rand(N, 3) * 2 - 1) * cfg.init_extent
    rgbs = torch.rand(N, 3) / 255.0 * SH_C0 + 0.5  # 3DGS: SH2RGB(rand/255) ~ grey
    means = means.to(device)
    dist = knn_mean_dist(means)
    scales = torch.log(dist * cfg.init_scale)[:, None].repeat(1, 3)
    quats = torch.rand(N, 4, device=device)
    opacities = torch.logit(torch.full((N,), cfg.init_opacity, device=device))
    K = (cfg.sh_degree + 1) ** 2
    colors = torch.zeros(N, K, 3, device=device)
    colors[:, 0, :] = (rgbs.to(device) - 0.5) / SH_C0
    params = [
        ("means", means, cfg.means_lr * scene_scale),
        ("scales", scales, cfg.scales_lr),
        ("quats", quats, cfg.quats_lr),
        ("opacities", opacities, cfg.opacities_lr),
        ("sh0", colors[:, :1, :], cfg.sh0_lr),
        ("shN", colors[:, 1:, :], cfg.shN_lr),
    ]
    splats = torch.nn.ParameterDict(
        {n: torch.nn.Parameter(v.contiguous()) for n, v, _ in params}
    )
    optimizers = {
        n: torch.optim.Adam([{"params": splats[n], "lr": lr, "name": n}], eps=1e-15)
        for n, _, lr in params
    }
    return splats, optimizers


def effective_attributes(splats, fld, planes=None, opacity_gate=False):
    """NEW: attr = attr_param + field(mu). Vanilla when fld is None.
    ``planes``: effective planes computed once this step (shared with the regulariser).
    ``opacity_gate``: scale the colour residual by sigmoid(stored opacity), detached."""
    sh0, opa = splats["sh0"], splats["opacities"]
    if fld is not None:
        res = fld(splats["means"], planes)  # detaches means unless grad_to_means
        if opacity_gate:
            sh0 = sh0 + torch.sigmoid(splats["opacities"]).detach()[:, None, None] * res["sh0"]
        else:
            sh0 = sh0 + res["sh0"]
        if "opacities" in res:
            opa = opa + res["opacities"]
    return {
        "means": splats["means"],
        "quats": splats["quats"],
        "scales": torch.exp(splats["scales"]),
        "opacities": torch.sigmoid(opa),
        "colors": torch.cat([sh0, splats["shN"]], 1),
    }


@torch.no_grad()
def field_residual_stats(splats, fld):
    """Mean |field sh0 residual| and its size relative to mean |stored sh0|.

    Re-queries the field at the CURRENT means: `attrs` from the forward pass is stale
    after the optimizer step and densification (different N), which crashed the first
    Checkpoint B launch."""
    d = fld(splats["means"])["sh0"].abs().mean()
    return d, d / splats["sh0"].abs().mean().clamp_min(1e-12)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    from gsplat import rasterization
    from gsplat.losses import create_ssim_window, torch_ssim_loss
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    from eval import evaluate

    cfg = parse_config()
    set_seed(cfg.seed)
    device = torch.device("cuda")
    tag = f"{cfg.variant}_{cfg.scene}_{cfg.seed}"
    os.makedirs(cfg.result_dir, exist_ok=True)
    print(f"[{tag}] config: {json.dumps(dataclasses.asdict(cfg))}", flush=True)

    train = load_split(cfg.data_dir, "train", cfg.white_bg)
    views = select_views(len(train["images"]), cfg.train_views)
    if len(views) < len(train["images"]):
        train = {"images": train["images"][views], "camtoworlds": train["camtoworlds"][views],
                 "K": train["K"], "names": [train["names"][i] for i in views]}
        print(f"[{tag}] sparse views: training on {len(views)} views {views}", flush=True)
    test = load_split(cfg.data_dir, "test", cfg.white_bg)
    train_imgs = train["images"].to(device)  # uint8, ~190 MB for 100 x 800^2
    H, W = train_imgs.shape[1:3]
    K = train["K"].to(device)
    train_viewmats = torch.linalg.inv(train["camtoworlds"]).to(device)
    test_viewmats = torch.linalg.inv(test["camtoworlds"]).to(device)
    scene_scale = scene_extent(train["camtoworlds"])
    bg = torch.ones(1, 3, device=device) if cfg.white_bg else torch.zeros(1, 3, device=device)
    print(f"[{tag}] {len(train_imgs)} train / {len(test['images'])} test views, "
          f"{W}x{H}, scene_scale={scene_scale:.3f}", flush=True)

    splats, optimizers = create_splats(cfg, scene_scale, device)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.iters)
    )
    if cfg.strategy == "default":
        strategy = DefaultStrategy(absgrad=cfg.absgrad)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    elif cfg.strategy == "mcmc":
        strategy = MCMCStrategy(cap_max=cfg.mcmc_cap_max)
        strategy_state = strategy.initialize_state()
    else:
        raise ValueError(cfg.strategy)
    strategy.check_sanity(splats, optimizers)

    # NEW: tri-plane field. It lives outside `splats`, so the densification
    # strategies never see it: new / relocated Gaussians simply query the field at
    # their new means on the next forward pass. No per-Gaussian field state exists.
    fld, fld_opt, fld_sched = None, None, None
    if cfg.field:
        fcfg = TriplaneFieldConfig(
            channels=cfg.field_channels,
            resolutions=(64, 128, 256) if cfg.field_multires else (cfg.field_res,),
            fusion=cfg.field_fusion,
            mlp_width=cfg.field_mlp_width,
            predict_opacity=cfg.field_opacity,
            plane_cnn=cfg.plane_cnn,
            cnn_off_after=cfg.plane_cnn_off_after,
            reg_on_raw=cfg.reg_on_raw,
            grad_to_means=cfg.field_grad_to_means,
        )
        # Build the field under a forked RNG: its random init must not consume the
        # global CPU RNG, otherwise the view order (torch.randperm below) would differ
        # from vanilla for the same seed. (Found in the pre-Checkpoint-B review.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 1_000_003)
            fld = TriplaneField(torch.tensor(cfg.field_aabb).view(2, 3), fcfg)
        fld = fld.to(device)
        fld_opt, fld_sched = make_field_optimizer(
            fld, cfg.plane_lr, cfg.cnn_lr, cfg.mlp_lr, cfg.iters
        )
        print(f"[{tag}] field params: {sum(p.numel() for p in fld.parameters()):,}", flush=True)

    ssim_window = create_ssim_window(11, 3, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    order: List[int] = []
    for step in range(cfg.iters):
        if not order:  # 3DGS: random views without replacement, refilled each epoch
            order = torch.randperm(len(train_imgs)).tolist()
        idx = order.pop()
        planes = None
        if fld is not None:
            fld.set_step(step)  # drives --plane_cnn_off_after
            planes = fld.effective_planes()  # CNN runs once per step

        attrs = effective_attributes(splats, fld, planes, cfg.field_opacity_gate)
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        renders, alphas, info = rasterization(
            **attrs,
            viewmats=train_viewmats[idx : idx + 1],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=sh_degree,
            packed=cfg.packed,
            absgrad=cfg.absgrad,
            backgrounds=bg,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
        )
        pred = renders[..., :3]  # [1, H, W, 3]
        gt = train_imgs[idx : idx + 1].float() / 255.0

        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)

        l1 = F.l1_loss(pred, gt)
        ssim_l = 1.0 - torch_ssim_loss(
            pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), ssim_window, 11, 3
        ).mean()
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * ssim_l
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * attrs["opacities"].mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * attrs["scales"].mean()
        reg = None
        if fld is not None:
            reg = fld.regularization(cfg.tv_weight, cfg.l1_weight, planes)  # post-CNN
            loss = loss + reg
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        if fld is not None:
            fld_opt.step()
            fld_opt.zero_grad(set_to_none=True)
            fld_sched.step()  # planes, CNN and MLP decay together

        if cfg.strategy == "default":
            strategy.step_post_backward(
                splats, optimizers, strategy_state, step, info, packed=cfg.packed
            )
        else:
            strategy.step_post_backward(
                splats, optimizers, strategy_state, step, info,
                lr=means_sched.get_last_lr()[0],
            )

        if step % cfg.log_every == 0 or step == cfg.iters - 1:
            with torch.no_grad():
                psnr = -10 * math.log10(F.mse_loss(pred.clamp(0, 1), gt).item())
            msg = (f"[{tag}] step {step:6d} loss {loss.item():.4f} train_psnr {psnr:.2f} "
                   f"N {len(splats['means']):,} "
                   f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                   f"t {(time.perf_counter() - t_start) / 60:.1f}min")
            if reg is not None:
                # Size of the field's colour residual relative to the stored sh0: shows
                # whether the field contributes at all (review note on the small mlp_lr).
                d_sh0, rel = field_residual_stats(splats, fld)
                msg += (f" reg {reg.item():.2e} field|dsh0| {d_sh0.item():.2e}"
                        f" ({rel.item():.1%}) cnn {'on' if fld.cnn_active else 'off'}")
            print(msg, flush=True)

    torch.cuda.synchronize()
    train_minutes = (time.perf_counter() - t_start) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 2**30

    # ---- evaluation: bake the field once (means are fixed at test time) ---------
    with torch.no_grad():
        if fld is not None:
            fld.set_step(cfg.iters)
        torch.cuda.synchronize()
        tq = time.perf_counter()
        baked = effective_attributes(splats, fld, opacity_gate=cfg.field_opacity_gate)
        torch.cuda.synchronize()
        field_query_ms = (time.perf_counter() - tq) * 1000.0

        def render_fn(i: int) -> torch.Tensor:
            out, _, _ = rasterization(
                **baked,
                viewmats=test_viewmats[i : i + 1],
                Ks=K[None],
                width=W,
                height=H,
                sh_degree=cfg.sh_degree,
                packed=cfg.packed,
                backgrounds=bg,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
            )
            return out[0, ..., :3]

        metrics = evaluate(
            render_fn,
            test["images"],
            device,
            render_dir=os.path.join(cfg.result_dir, "renders", tag),
            save_indices=cfg.save_render_indices,
        )

    result = {
        "variant": cfg.variant,
        "scene": cfg.scene,
        "seed": cfg.seed,
        "git_commit": git_commit(),
        "iters": cfg.iters,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips_vgg": metrics["lpips_vgg"],
        "lpips_alex": metrics["lpips_alex"],
        "fps": metrics["fps"],
        "num_gaussians": int(len(splats["means"])),
        "peak_vram_gb": peak_vram_gb,
        "train_minutes": train_minutes,
        "gpu_name": torch.cuda.get_device_name(device),
        "config": dataclasses.asdict(cfg),
        # extras (not in the required schema, but cheap and useful)
        "extra": {
            "peak_vram_reserved_gb": peak_reserved_gb,
            "field_query_ms": field_query_ms,
            "num_test_views": metrics["num_test_views"],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gsplat": __import__("gsplat").__version__,
        },
    }
    if cfg.save_gaussians:
      try:  # diagnostics must never cost us the run's metrics JSON
        with torch.no_grad():
            out = {
                "means": splats["means"],
                "opacity": torch.sigmoid(splats["opacities"]),
                "sh0": splats["sh0"][:, 0],
                "scale_max": torch.exp(splats["scales"]).max(-1).values,
            }
            if fld is not None:
                out["field_sh0"] = fld(splats["means"])["sh0"][:, 0]  # ungated residual
            gdir = os.path.join(cfg.result_dir, "gaussians")
            os.makedirs(gdir, exist_ok=True)
            np.savez_compressed(
                os.path.join(gdir, f"{tag}.npz"),
                **{k: v.float().cpu().numpy().astype(np.float16) for k, v in out.items()},
            )
      except Exception as ex:
        print(f"[{tag}] WARNING: save_gaussians failed: {ex!r}", flush=True)

    path = os.path.join(cfg.result_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] PSNR {result['psnr']:.3f} SSIM {result['ssim']:.4f} "
          f"LPIPS-VGG {result['lpips_vgg']:.4f} LPIPS-Alex {result['lpips_alex']:.4f} "
          f"FPS {result['fps']:.1f} N {result['num_gaussians']:,} "
          f"VRAM {peak_vram_gb:.2f}GB train {train_minutes:.1f}min -> {path}", flush=True)


if __name__ == "__main__":
    main()
