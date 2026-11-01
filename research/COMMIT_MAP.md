# Commit ID map (history rewrite, 2026-11-01)

The branch history was rewritten to set the author to adk7712 and remove tool co-author lines.
File contents are unchanged. Result JSONs keep the commit IDs they were run from (left column);
the same code is at the new ID (right column). `kaggle/configs/*.json` refs were updated.

| old commit | new commit | subject |
|---|---|---|
| 401bda29033e | 04091805fa24 | kaggle: 12-view sparse test config (3 variants x 8 scenes, seed 0) |
| 6b93dbb400c5 | a1ef3385a1cb | research: --train_views for sparse-view training (evenly spaced subset of train views) |
| 9a88f124a196 | 30dc60f7842d | research: opacity-gate test results and ghost diagnostics (seed 0, 6 scenes) |
| 060731756bcc | 7803ffa45132 | research: ghost-hypothesis analysis of saved Gaussians (visual-hull inside/outside/below) |
| c6880e7f8b7d | b36987a6d0bc | kaggle: config for ghost diagnostics + opacity-gate test (16 runs) |
| 76c921425d10 | 27c0a045c244 | research: opacity-gated field residual and save_gaussians diagnostics |
| beae91e4a591 | e60c59516344 | kaggle: point checkpoint configs at the triplane-field branch |
| 9bec281dc008 | 25b0414c4948 | Phase 1 wrap-up: commit 96 result JSONs, final tables in README, record pip freeze per run |
| 3bc3be712cbc | 972f3aada935 | research: verify 3DGS init range and LPIPS input handling against source code |
| 060c332e6b2d | 933da7007861 | research: verify 3DGS NeRF-Synthetic reference numbers against the paper |
| 73e9200ff84f | b10b57c8252a | research: mark unverified claims in README and CNN report |
| 805b22f7930d | f0283460e1ff | research: analysis of why the plane CNN helps some scenes and hurts others |
| f486d47e7c76 | 0f92192a3c5b | kaggle: point sweep configs at the channels-first field (13c58fc) |
| 0238025d8fbb | 876891f91d00 | kaggle: speed-up verification config |
| 13c58fcff848 | 87aa35a6e3a7 | research: channels-first field forward (removes strided copies); drop embedding sampler |
| f483aa8cada5 | d591d71d15b4 | research: document field-overhead profiling results |
| 90cc4d1c3f21 | 6c35552d2492 | kaggle: overhead profiling job config |
| 1a4b9bc9641d | 50ed5c57da99 | research: embedding_bag plane sampler + field micro-benchmark; runner per-job args |
| 47c1d4138657 | 3c134c3227c2 | kaggle: Phase 1 sweep configs (one per seed) and DATASET_OWNER for teammates |
| ff897337416f | 7f3c99192d30 | research: fix field-residual log line crashing after densification |
| e61919f58196 | bb466e3a63d0 | research: fixes from pre-Checkpoint-B review |
| e1feef420174 | 3e16e056a125 | research: record pinned Kaggle versions and Checkpoint A baseline |
| 8742c2061e5f | 823042d0d558 | kaggle: accept KGAT_ tokens via KAGGLE_API_TOKEN; document dataset provenance |
| d8281fcaa2f7 | d1483f6262f4 | research: NeRF-Synthetic trainer, eval protocol, aggregate, variant configs, README |
| 223f6224bb61 | 35ceed3833cc | research: tri-plane attribute field with residual plane CNN + CPU tests |
| abbd399cd4f8 | f132de37648f | kaggle: T4 script-kernel runner with launch/status/fetch scripts |
