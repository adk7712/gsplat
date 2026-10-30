"""Tri-plane attribute field for static 3DGS (research addition, not upstream gsplat).

Every class/function in this file is NEW code for the MLDA x EEE tri-plane study.
Nothing here modifies gsplat; the trainer (research/train.py) adds the field's
output to the per-Gaussian parameters:

    attr = attr_param + field(mu)

Design (see research/README.md for the reasoning behind each choice):
  * Three learnable planes (xy, xz, yz), each C x R x R, optionally at several
    resolutions (e.g. {64, 128, 256}).
  * Gaussian means are normalised into the scene AABB -> [-1, 1] and the planes are
    bilinearly sampled with ``grid_sample(align_corners=True)``.
  * Planes are fused by elementwise product (K-Planes style) or concatenation
    (flag), then concatenated across resolutions.
  * A 2-layer MLP (width 64) maps the feature to residuals for SH DC colour and
    (optionally) the opacity logit. The last MLP layer is zero-initialised so the
    field outputs exactly 0 at step 0, i.e. training starts at vanilla gsplat.
  * Optional plane CNN, ported from our 4DGaussians HexPlane experiment:
        plane_eff = plane + conv2(relu(conv1(plane)))
    3x3 convs, padding 1, channels preserved, conv2 zero-initialised -> identity at
    init. ``cnn_off_after`` bypasses the CNN (uses raw planes) after a given step.
  * TV / L1 regularisers act on the EFFECTIVE (post-CNN) planes by default.
    FIX vs. our 4D code, which regularised the raw parameters only, leaving the
    CNN output unconstrained. ``reg_on_raw=True`` restores the old behaviour for
    an ablation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Axis pairs for the three planes. Plane k is indexed by coordinates
# (PLANE_AXES[k][0], PLANE_AXES[k][1]); grid_sample expects (x, y) = (width, height),
# so the first axis indexes the plane's W dimension and the second its H dimension.
PLANE_AXES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))  # xy, xz, yz
PLANE_NAMES = ("xy", "xz", "yz")


@dataclass
class TriplaneFieldConfig:
    """Hyper-parameters of the field. All defaults follow the Phase 1 plan."""

    channels: int = 32  # C
    resolutions: Sequence[int] = (128,)  # (128,) single-res; (64, 128, 256) multi-res
    fusion: str = "product"  # "product" (K-Planes) or "concat"
    mlp_width: int = 64
    mlp_layers: int = 2  # number of Linear layers (hidden + output)
    predict_opacity: bool = False  # also output an opacity-logit residual
    plane_cnn: bool = False  # enable the residual plane CNN
    cnn_off_after: int = -1  # bypass the CNN for step >= this (-1 = never)
    reg_on_raw: bool = False  # ablation: regularise raw planes instead of effective
    # Plane init range. Product fusion needs non-zero planes (a zero factor kills
    # the feature and its gradient), so we follow K-Planes: U(0.1, 0.5).
    init_low: float = 0.1
    init_high: float = 0.5


class PlaneCNN(nn.Module):
    """Residual 2-layer CNN applied to ONE feature plane (ported from our 4D code).

    plane_eff = plane + conv2(relu(conv1(plane)))

    conv2 weight AND bias are zero-initialised, so the module is exactly the
    identity at init; conv1 keeps PyTorch's default init so that gradients reach
    conv2 from the first step.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        # plane: [1, C, H, W]
        return plane + self.conv2(F.relu(self.conv1(plane)))


class TriplaneField(nn.Module):
    """Tri-plane field queried at Gaussian centres -> per-Gaussian residuals.

    Args:
        aabb: [2, 3] tensor (min corner, max corner) of the scene box used to
            normalise means. Means outside the box are clamped to the border
            (``padding_mode="border"``) rather than zeroed: with product fusion a
            zero feature would silently disconnect those Gaussians from the field.
        cfg: TriplaneFieldConfig.
    """

    def __init__(self, aabb: torch.Tensor, cfg: TriplaneFieldConfig):
        super().__init__()
        assert cfg.fusion in ("product", "concat"), cfg.fusion
        self.cfg = cfg
        self.register_buffer("aabb", aabb.detach().clone().float().reshape(2, 3))

        C = cfg.channels
        # planes[r][k]: Parameter [1, C, R, R] for resolution r and plane k.
        self.planes = nn.ParameterList()
        for R in cfg.resolutions:
            for _ in PLANE_AXES:
                p = torch.empty(1, C, R, R).uniform_(cfg.init_low, cfg.init_high)
                self.planes.append(nn.Parameter(p))

        # One CNN per (resolution, plane): planes at different axes/scales hold
        # different content, so weights are not shared.
        self.cnns = (
            nn.ModuleList([PlaneCNN(C) for _ in self.planes]) if cfg.plane_cnn else None
        )

        n_res = len(cfg.resolutions)
        feat_dim = C * n_res if cfg.fusion == "product" else 3 * C * n_res
        self.out_dim = 3 + (1 if cfg.predict_opacity else 0)
        layers: List[nn.Module] = []
        d = feat_dim
        for _ in range(cfg.mlp_layers - 1):
            layers += [nn.Linear(d, cfg.mlp_width), nn.ReLU(inplace=True)]
            d = cfg.mlp_width
        last = nn.Linear(d, self.out_dim)
        # Zero-init of the last layer: field(mu) == 0 at step 0 -> vanilla gsplat.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

        # Current training iteration, set by the trainer every step; used only for
        # the cnn_off_after bypass.
        self._step = 0

    # ------------------------------------------------------------------ CNN bypass
    def set_step(self, step: int) -> None:
        self._step = int(step)

    @property
    def cnn_active(self) -> bool:
        if self.cnns is None:
            return False
        if self.cfg.cnn_off_after >= 0 and self._step >= self.cfg.cnn_off_after:
            return False  # --plane_cnn_off_after: use raw planes from here on
        return True

    # ------------------------------------------------------------------ planes
    def effective_planes(self) -> List[torch.Tensor]:
        """Planes as seen by the sampler: post-CNN if the CNN is active, else raw."""
        if not self.cnn_active:
            return list(self.planes)
        return [cnn(p) for cnn, p in zip(self.cnns, self.planes)]

    def normalize(self, means: torch.Tensor) -> torch.Tensor:
        lo, hi = self.aabb[0], self.aabb[1]
        return (means - lo) / (hi - lo) * 2.0 - 1.0  # [N, 3] in [-1, 1] inside box

    def features(
        self, means: torch.Tensor, planes: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if planes is None:
            planes = self.effective_planes()
        x = self.normalize(means)
        n_planes = len(PLANE_AXES)
        feats = []
        for r in range(len(self.cfg.resolutions)):
            per_plane = []
            for k, (a, b) in enumerate(PLANE_AXES):
                grid = x[:, [a, b]].view(1, -1, 1, 2)  # [1, N, 1, 2]
                f = F.grid_sample(
                    planes[r * n_planes + k],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )  # [1, C, N, 1]
                per_plane.append(f[0, :, :, 0].t())  # [N, C]
            if self.cfg.fusion == "product":
                feats.append(per_plane[0] * per_plane[1] * per_plane[2])
            else:
                feats.append(torch.cat(per_plane, dim=-1))
        return torch.cat(feats, dim=-1)  # concatenate across resolutions

    def forward(self, means: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns residuals: {"sh0": [N, 1, 3], "opacities": [N] (if enabled)}."""
        out = self.mlp(self.features(means))
        res = {"sh0": out[:, :3].unsqueeze(1)}
        if self.cfg.predict_opacity:
            res["opacities"] = out[:, 3]
        return res

    # ------------------------------------------------------------------ regularisers
    def _reg_planes(self) -> List[torch.Tensor]:
        # FIX vs. our 4D code: regularise what the sampler actually sees (post-CNN).
        return list(self.planes) if self.cfg.reg_on_raw else self.effective_planes()

    @staticmethod
    def tv(planes: List[torch.Tensor]) -> torch.Tensor:
        """K-Planes-style TV: mean squared difference of neighbours, averaged over planes."""
        total = 0.0
        for p in planes:
            dh = (p[..., 1:, :] - p[..., :-1, :]).pow(2).mean()
            dw = (p[..., :, 1:] - p[..., :, :-1]).pow(2).mean()
            total = total + dh + dw
        return total / len(planes)

    @staticmethod
    def l1(planes: List[torch.Tensor]) -> torch.Tensor:
        return sum(p.abs().mean() for p in planes) / len(planes)

    def regularization(self, tv_weight: float, l1_weight: float) -> torch.Tensor:
        if tv_weight == 0 and l1_weight == 0:
            return self.planes[0].new_zeros(())
        planes = self._reg_planes()
        loss = self.planes[0].new_zeros(())
        if tv_weight > 0:
            loss = loss + tv_weight * self.tv(planes)
        if l1_weight > 0:
            loss = loss + l1_weight * self.l1(planes)
        return loss

    # ------------------------------------------------------------------ optimiser
    def param_groups(
        self, plane_lr: float, cnn_lr: float, mlp_lr: float
    ) -> List[Dict]:
        """Separate initial lrs; the trainer puts all groups in ONE Adam with ONE
        ExponentialLR, so planes, CNN and MLP share the same decay schedule.
        (FIX vs. our 4D code, where the CNN silently stayed at a constant lr.)"""
        groups = [
            {"params": list(self.planes), "lr": plane_lr, "name": "planes"},
            {"params": list(self.mlp.parameters()), "lr": mlp_lr, "name": "mlp"},
        ]
        if self.cnns is not None:
            groups.append(
                {"params": list(self.cnns.parameters()), "lr": cnn_lr, "name": "cnn"}
            )
        return groups


def make_field_optimizer(
    fld: TriplaneField,
    plane_lr: float,
    cnn_lr: float,
    mlp_lr: float,
    max_steps: int,
    final_lr_ratio: float = 0.01,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """One Adam + one ExponentialLR over all field groups (same schedule shape as
    gsplat's means lr: decays to ``final_lr_ratio`` x initial by ``max_steps``)."""
    opt = torch.optim.Adam(fld.param_groups(plane_lr, cnn_lr, mlp_lr), eps=1e-15)
    gamma = final_lr_ratio ** (1.0 / max_steps)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    return opt, sched
