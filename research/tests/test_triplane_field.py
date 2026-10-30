"""CPU-only tests for research/triplane_field.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

import triplane_field as tf
from triplane_field import (
    PlaneCNN,
    TriplaneField,
    TriplaneFieldConfig,
    make_field_optimizer,
)

AABB = torch.tensor([[-1.5] * 3, [1.5] * 3])


def make(seed=0, **kw):
    kw.setdefault("channels", 4)
    kw.setdefault("resolutions", (8,))
    kw.setdefault("mlp_width", 16)
    torch.manual_seed(seed)
    return TriplaneField(AABB, TriplaneFieldConfig(**kw))


def rand_means(n=32, lim=1.4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=g) * 2 - 1) * lim


def perturb_last_mlp(fld, std=0.1):
    with torch.no_grad():
        fld.mlp[-1].weight.normal_(0, std)
        fld.mlp[-1].bias.normal_(0, std)


def perturb_conv2(fld, std=0.1):
    with torch.no_grad():
        for c in fld.cnns:
            c.conv2.weight.normal_(0, std)
            c.conv2.bias.normal_(0, std)


# ------------------------------------------------------------------ shapes
@pytest.mark.parametrize("fusion", ["product", "concat"])
@pytest.mark.parametrize("res", [(8,), (4, 8)])
@pytest.mark.parametrize("opa", [False, True])
def test_output_shape(fusion, res, opa):
    fld = make(fusion=fusion, resolutions=res, predict_opacity=opa)
    out = fld(rand_means(17))
    assert out["sh0"].shape == (17, 1, 3)
    if opa:
        assert out["opacities"].shape == (17,)
    else:
        assert "opacities" not in out
    assert set(out) == ({"sh0", "opacities"} if opa else {"sh0"})


# ------------------------------------------------------------------ zero init
@pytest.mark.parametrize("cnn", [False, True])
@pytest.mark.parametrize("fusion", ["product", "concat"])
def test_zero_init(cnn, fusion):
    fld = make(plane_cnn=cnn, fusion=fusion, predict_opacity=True, resolutions=(4, 8))
    out = fld(rand_means())
    assert torch.equal(out["sh0"], torch.zeros_like(out["sh0"]))
    assert torch.equal(out["opacities"], torch.zeros_like(out["opacities"]))


# ------------------------------------------------------------------ CNN identity
def test_plane_cnn_identity():
    torch.manual_seed(0)
    cnn = PlaneCNN(4)
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(cnn(x), x)


def test_effective_planes_equal_raw_at_init():
    fld = make(plane_cnn=True, resolutions=(4, 8))
    eff = fld.effective_planes()
    assert len(eff) == len(fld.planes) == 6
    for e, p in zip(eff, fld.planes):
        assert torch.equal(e, p)


# ------------------------------------------------------------------ gradients
@pytest.mark.parametrize("opa", [False, True])
def test_gradients_reach_everything(opa):
    fld = make(plane_cnn=True, predict_opacity=opa)
    means = rand_means(64)
    target = torch.randn(64, 1, 3)
    opt, _ = make_field_optimizer(fld, 1e-2, 1e-2, 1e-2, max_steps=100)

    def loss_fn():
        return (fld(means)["sh0"] - target).pow(2).mean()

    # step 0: only the last MLP layer (and conv2 -> no, conv2 is behind a zero
    # last layer) gets non-zero gradient.
    loss_fn().backward()
    last = fld.mlp[-1]
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0
    opt.step()
    opt.zero_grad()

    for _ in range(2):
        loss_fn().backward()
        opt.step()
        opt.zero_grad()

    loss_fn().backward()
    named = [(f"planes.{i}", p) for i, p in enumerate(fld.planes)]
    for i, c in enumerate(fld.cnns):
        named += [(f"cnn.{i}.conv1.weight", c.conv1.weight),
                  (f"cnn.{i}.conv2.weight", c.conv2.weight)]
    named += list(fld.mlp.named_parameters())
    for name, p in named:
        if name.startswith("planes") or "weight" in name:
            assert p.grad is not None, name
            assert p.grad.abs().sum() > 0, name
    # MLP biases too (opacity column bias may be fine either way; only check sh0 rows)
    for name, p in fld.mlp.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


# ------------------------------------------------------------------ regularisers
def test_tv_on_effective_planes():
    fld = make(plane_cnn=True, resolutions=(4, 8))
    perturb_conv2(fld)
    eff = fld.effective_planes()
    assert not torch.allclose(eff[0], fld.planes[0])
    reg = fld.regularization(tv_weight=1.0, l1_weight=0.0)
    assert torch.allclose(reg, TriplaneField.tv(eff))
    assert not torch.allclose(reg, TriplaneField.tv(list(fld.planes)))


def test_tv_reg_on_raw():
    fld = make(plane_cnn=True, reg_on_raw=True)
    perturb_conv2(fld)
    reg = fld.regularization(1.0, 0.0)
    assert torch.allclose(reg, TriplaneField.tv(list(fld.planes)))
    assert not torch.allclose(reg, TriplaneField.tv(fld.effective_planes()))


def test_l1_on_effective_and_raw():
    fld = make(plane_cnn=True)
    perturb_conv2(fld)
    reg = fld.regularization(0.0, 1.0)
    assert torch.allclose(reg, TriplaneField.l1(fld.effective_planes()))
    assert not torch.allclose(reg, TriplaneField.l1(list(fld.planes)))
    fld.cfg.reg_on_raw = True
    assert torch.allclose(fld.regularization(0.0, 1.0), TriplaneField.l1(list(fld.planes)))


def test_zero_weights_give_zero_reg():
    fld = make(plane_cnn=True)
    r = fld.regularization(0.0, 0.0)
    assert r.shape == () and r.item() == 0.0


def _cnn_grad_mass(fld):
    tot = 0.0
    for p in fld.cnns.parameters():
        if p.grad is not None:
            tot += p.grad.abs().sum().item()
    return tot


@pytest.mark.parametrize("which", ["tv", "l1"])
@pytest.mark.parametrize("raw", [False, True])
def test_reg_grads_on_cnn(which, raw):
    fld = make(plane_cnn=True, reg_on_raw=raw)
    perturb_conv2(fld)
    w = (1.0, 0.0) if which == "tv" else (0.0, 1.0)
    fld.regularization(*w).backward()
    # raw planes always get a gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fld.planes)
    conv2_mass = sum(c.conv2.weight.grad.abs().sum().item()
                     for c in fld.cnns if c.conv2.weight.grad is not None)
    if raw:
        assert _cnn_grad_mass(fld) == 0.0
    else:
        assert conv2_mass > 0.0


# ------------------------------------------------------------------ cnn_off_after
def test_cnn_off_after():
    fld = make(plane_cnn=True, cnn_off_after=10)
    perturb_conv2(fld)
    perturb_last_mlp(fld)
    means = rand_means()
    raw_out = fld.mlp(fld.features(means, planes=list(fld.planes)))[:, :3].unsqueeze(1)

    fld.set_step(9)
    assert fld.cnn_active
    assert not torch.allclose(fld(means)["sh0"], raw_out)

    fld.set_step(10)
    assert not fld.cnn_active
    eff = fld.effective_planes()
    assert len(eff) == len(fld.planes)
    for e, p in zip(eff, fld.planes):
        assert e is p
    assert torch.allclose(fld(means)["sh0"], raw_out, atol=1e-6)

    fld.zero_grad()
    fld(means)["sh0"].sum().backward()
    for p in fld.cnns.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fld.planes)


def test_cnn_never_off():
    fld = make(plane_cnn=True, cnn_off_after=-1)
    fld.set_step(10**6)
    assert fld.cnn_active
    assert not make(plane_cnn=False).cnn_active


# ------------------------------------------------------------------ LR schedule
def test_lr_schedule():
    fld = make(plane_cnn=True)
    lrs = {"planes": 3e-3, "mlp": 5e-4, "cnn": 7e-5}
    opt, sched = make_field_optimizer(fld, lrs["planes"], lrs["cnn"], lrs["mlp"], max_steps=100)
    assert {g["name"] for g in opt.param_groups} == set(lrs)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(lrs[g["name"]])
    assert len({g["lr"] for g in opt.param_groups}) == 3
    for _ in range(100):
        opt.step()
        sched.step()
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(0.01 * lrs[g["name"]], rel=1e-4)


def test_no_cnn_group_without_cnn():
    fld = make(plane_cnn=False)
    opt, _ = make_field_optimizer(fld, 1e-2, 1e-2, 1e-2, max_steps=10)
    assert [g["name"] for g in opt.param_groups] == ["planes", "mlp"]


# ------------------------------------------------------------------ out of box
@pytest.mark.parametrize("fusion", ["product", "concat"])
def test_outside_aabb_border_clamped(fusion):
    fld = make(fusion=fusion, resolutions=(4, 8))
    perturb_last_mlp(fld)
    far = torch.tensor([[5.0, -7.0, 2.0], [100.0, 0.3, -50.0], [-1.6, 1.51, 0.0]])
    clamped = far.clamp(-1.5, 1.5)
    f_far = fld.features(far)
    f_cl = fld.features(clamped)
    assert torch.isfinite(f_far).all()
    assert torch.allclose(f_far, f_cl, atol=1e-6)
    out = fld(far)["sh0"]
    assert torch.isfinite(out).all()
    assert torch.allclose(out, fld(clamped)["sh0"], atol=1e-6)


# --- Added after the pre-Checkpoint-B review --------------------------------------
def test_field_construction_under_fork_rng_preserves_view_order():
    """train.py builds the field under fork_rng so the view order matches vanilla."""
    torch.manual_seed(0)
    ref = torch.randperm(100)
    torch.manual_seed(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1_000_003)
        TriplaneField(_AABB_FOR_RNG_TEST, TriplaneFieldConfig(plane_cnn=True))
    assert torch.equal(torch.randperm(100), ref)


def test_means_detached_unless_grad_to_means():
    for flag in (False, True):
        fld = TriplaneField(
            _AABB_FOR_RNG_TEST, TriplaneFieldConfig(channels=4, resolutions=(8,), grad_to_means=flag)
        )
        torch.nn.init.normal_(fld.mlp[-1].weight, std=0.1)
        means = (torch.rand(50, 3) * 2 - 1).requires_grad_()
        fld(means)["sh0"].sum().backward()
        assert (means.grad is not None and means.grad.abs().sum() > 0) == flag


def test_shared_planes_match_internal_planes():
    fld = TriplaneField(_AABB_FOR_RNG_TEST, TriplaneFieldConfig(channels=4, resolutions=(8,), plane_cnn=True))
    for c in fld.cnns:
        torch.nn.init.normal_(c.conv2.weight, std=0.1)
    torch.nn.init.normal_(fld.mlp[-1].weight, std=0.1)
    means = torch.rand(50, 3) * 2 - 1
    planes = fld.effective_planes()
    assert torch.allclose(fld(means, planes)["sh0"], fld(means)["sh0"])
    assert torch.allclose(fld.regularization(1.0, 0.1, planes), fld.regularization(1.0, 0.1))


_AABB_FOR_RNG_TEST = torch.tensor([[-1.5] * 3, [1.5] * 3])


def test_field_residual_stats_after_densification():
    """Regression: the trainer's log line must work after N changes (densification)."""
    import train

    fld = TriplaneField(_AABB_FOR_RNG_TEST, TriplaneFieldConfig(channels=4, resolutions=(8,)))
    splats = {"means": torch.rand(30, 3), "sh0": torch.rand(30, 1, 3) + 0.1}
    d, rel = train.field_residual_stats(splats, fld)
    assert d.item() == 0.0 and rel.item() == 0.0  # zero-init field
    torch.nn.init.normal_(fld.mlp[-1].weight, std=0.1)
    splats = {"means": torch.rand(45, 3), "sh0": torch.rand(45, 1, 3) + 0.1}  # N grew
    d, rel = train.field_residual_stats(splats, fld)
    assert d.item() > 0 and torch.isfinite(rel)


# --- Added with the channels-first layout optimisation ----------------------------
@pytest.mark.parametrize("res", [(8,), (4, 8, 16)])
@pytest.mark.parametrize("fusion", ["product", "concat"])
@pytest.mark.parametrize("opacity", [False, True])
def test_optimised_forward_matches_reference(res, fusion, opacity):
    """forward() (channels-first) == forward_reference() (original code): outputs and
    gradients w.r.t. planes, CNN, MLP and means."""
    torch.manual_seed(0)
    fld = TriplaneField(
        _AABB_FOR_RNG_TEST,
        TriplaneFieldConfig(channels=4, resolutions=res, fusion=fusion, plane_cnn=True,
                            grad_to_means=True, predict_opacity=opacity),
    )
    for c in fld.cnns:
        torch.nn.init.normal_(c.conv2.weight, std=0.1)
    torch.nn.init.normal_(fld.mlp[-1].weight, std=0.1)
    torch.nn.init.normal_(fld.mlp[-1].bias, std=0.1)
    torch.manual_seed(1)
    means0 = torch.cat([torch.rand(200, 3) * 3 - 1.5, torch.randn(20, 3) * 3])
    results = {}
    for name in ("forward", "forward_reference"):
        fld.zero_grad(set_to_none=True)
        means = means0.clone().requires_grad_()
        out = getattr(fld, name)(means)
        loss = (out["sh0"] * torch.linspace(-1, 1, out["sh0"].numel()).view_as(out["sh0"])).sum()
        if opacity:
            loss = loss + out["opacities"].pow(2).sum()
        loss.backward()
        results[name] = (out, {n: p.grad.clone() for n, p in fld.named_parameters()}, means.grad.clone())
    (o1, g1, m1), (o2, g2, m2) = results["forward"], results["forward_reference"]
    for k in o1:
        assert o1[k].shape == o2[k].shape and torch.allclose(o1[k], o2[k], atol=1e-6), k
    for n in g1:
        assert torch.allclose(g1[n], g2[n], atol=1e-5, rtol=1e-4), n
    assert torch.allclose(m1, m2, atol=1e-5, rtol=1e-4)
