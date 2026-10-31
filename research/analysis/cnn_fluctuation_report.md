# Why the plane CNN helps some scenes and hurts others (Phase 1, NeRF-Synthetic)

Data: 96 runs (4 variants × 8 scenes × 3 seeds, commit 13c58fc). "CNN effect" = PSNR(triplane_cnn) − PSNR(triplane).

| Scene | CNN vs no-CNN | CNN vs vanilla | Share of CNN's extra error in empty background | Field colour use, no CNN → CNN |
|---|---|---|---|---|
| mic | **+0.18** | +0.16 | – | 30% → 14% |
| hotdog | +0.06 | +0.07 | – | 28% → 16% |
| lego | +0.02 | +0.05 | – | 45% → 26% |
| drums | +0.01 | +0.02 | – | 29% → 44% |
| ficus | −0.01 | 0.00 | – | 33% → 60% |
| materials | −0.05 | −0.04 | – | 46% → 36% |
| ship | −0.05 | −0.11 | **95%** | 87% → 116% |
| chair | −0.09 | −0.18 | **97%** | 21% → 43% |

"Field colour use" = mean |field SH-DC residual| / mean |stored SH-DC| at the end of training.

## Hypotheses

**H1: the CNN tints "ghost" blobs in empty space (supported).** In chair and ship, 95–97% of the CNN's
extra error sits in the white background, not on the object: a smudge under the chair seat and under
the ship's base (`cnn_vs_nocnn.png`). **Unverified:** that these smudges are made of stray low-opacity
Gaussians, and that these regions are poorly constrained because cameras see them only at grazing angles.
No Gaussian positions or opacities were saved, and the only camera check done is that no training camera
sits below these objects. The field colours every Gaussian from the
three axis-aligned planes, and blobs directly under the object share the top-view plane cells with the
object above. With the CNN smoothing the planes, they inherit the object's colour (green under the ship,
dark under the seat).

**H2: where the CNN helps, it makes the field more conservative (supported).** The CNN roughly halves
how much colour the field carries in mic, hotdog and lego, and slightly lowers error on surfaces and
edges. Where it hurts (chair, ship, ficus, drums), it increases field use. Rank correlation between
"change in field use" and CNN effect over the 8 scenes: ρ = −0.76 (p = 0.04, permutation test).

**H3: the CNN blurs fine detail (rejected).** On object edges the CNN is equal or slightly better in
chair and ship. Image detail vs CNN effect: ρ = 0.22.

**Not related (no evidence):** object size / share of the planes used (ρ = −0.14), change in number of
Gaussians (ρ = −0.10), camera coverage below the object (ρ = +0.29), seed-to-seed spread (ρ = −0.24).
**Inconclusive:** overlapping surfaces in the plane projections (ρ = −0.57, p = 0.15; materials has by
far the most overlap and is slightly hurt).

Unverified (from memory or visual reading, not measured): the scene descriptions (chair slats, ship
rigging, mic mesh). Caveats: 8 scenes, 4 saved test views per run, about 8 tests run, so p = 0.04 is suggestive only;
all evidence is correlational.

## What this implies
- The CNN's losses come from colouring floaters, not from late over-smoothing, so the
  "switch the CNN off late" idea targets the wrong cause.
- Fixes that target H1: apply the field residual only to sufficiently opaque Gaussians (or scale it by
  opacity), or penalise the size of the field residual.
- Direct test of H1 (cheap): one short run per scene that saves Gaussian positions and opacities, to
  check that the tinted blobs are low-opacity floaters under the object.

Scripts: `scene_props.py` (visual hull, plane coverage, overlap), `render_err.py` (error split by
pixel type), `figure.py` (figure).
