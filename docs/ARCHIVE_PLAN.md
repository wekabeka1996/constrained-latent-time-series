# Archive Plan

**Phase:** 1P  
**Date:** 2026-06-20  
**Status:** EXECUTED (Phase 1Q Cleanup Completed)

> [!CAUTION]
> Do not execute any moves in this plan until the user explicitly approves
> Phase 1Q (Cleanup Patch). This document is a planning artefact only.

---

## Proposed Archive Structure

```
archive/
├── README_ARCHIVE.md              ← Explains this folder's purpose
├── legacy_code/                   ← Standalone scripts not part of the runner
├── invalidated_geometry/          ← Files from the P0 Jacobian bug
├── legacy_results/                ← Experimental outputs from generation_1/2
├── original_research/             ← Original Ukrainian research article
├── old_experiments/               ← Experiment scripts from src/experiments/
├── orphan_artifacts/              ← Root-level and data/generated/ pollution
└── legacy_models/                 ← Beta-sweep and architecture VAE weights
```

---

## archive/README_ARCHIVE.md (Draft Content)

```markdown
# Archive

This folder contains legacy exploratory code, invalidated artifacts, and
historical outputs retained for audit history only.

These files are NOT part of the clean reproduction pipeline defined in
`src/reproduce.py`.

## Why these files are retained

- To preserve audit history for the claim-by-claim verdict in
  `docs/CLAIM_VERDICT_MATRIX.md`.
- To provide provenance for the invalidated Fisher metric outputs.
- To retain the original research article for citation purposes.
- To keep experimental code reachable for future reference.

## What is NOT safe to cite from this archive

- `invalidated_geometry/` — Fisher metric outputs are mathematically
  invalid due to a Jacobian bug. See `docs/GEOMETRY_VALIDITY_STATUS.md`.
- `original_research/Стаття_Результат_Дослідження.md` — The article
  contains claims that were partially or fully invalidated by the clean
  reproduction. See `docs/CLAIM_VERDICT_MATRIX.md`.
- `legacy_results/generation_1/fisher_metrics/` — Same as above.
- `orphan_artifacts/generated_valid_thetas.npy` — A forbidden artifact
  from the legacy generator. Do not use as evidence.

## Integrity policy

Moving a file to this archive does NOT delete it from git history.
All moves must be accompanied by a git commit message explaining the reason.
```

---

## Proposed Moves

### Group 1: Invalidated Geometry Code

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `src/compute_fisher_metric.py` | `archive/invalidated_geometry/compute_fisher_metric.py` | P0 Jacobian bug; invalidated | LOW | `python -m pytest tests -q` |
| `src/regularized_fisher_metric.py` | `archive/invalidated_geometry/regularized_fisher_metric.py` | Depends on invalidated code | LOW | `python -m pytest tests -q` |
| `src/analyze_fisher_metrics.py` | `archive/invalidated_geometry/analyze_fisher_metrics.py` | Loads invalidated `.npy` files | LOW | `python -m pytest tests -q` |

### Group 2: Legacy Code (Standalone Scripts)

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `src/analyze_interpolation_results.py` | `archive/legacy_code/analyze_interpolation_results.py` | Hardcoded `results/generation_1/` | LOW | `python -m pytest tests -q` |
| `src/analyze_interpolation_norm_vs_validity.py` | `archive/legacy_code/analyze_interpolation_norm_vs_validity.py` | Same | LOW | `python -m pytest tests -q` |
| `src/analyze_roots_boundary_validity.py` | `archive/legacy_code/analyze_roots_boundary_validity.py` | Same | LOW | `python -m pytest tests -q` |
| `src/baseline_random_search.py` | `archive/legacy_code/baseline_random_search.py` | Imports from `src.Симуляція_2` | LOW | `python -m pytest tests -q` |
| `src/generate_latent_vectors.py` | `archive/legacy_code/generate_latent_vectors.py` | E402 import; standalone only | LOW | `python -m pytest tests -q` |
| `src/latent_space_interpolation_analysis.py` | `archive/legacy_code/latent_space_interpolation_analysis.py` | Logic ported into `validation.py` | LOW | `python -m pytest tests -q` |
| `src/latent_space_visualization.py` | `archive/legacy_code/latent_space_visualization.py` | Legacy t-SNE/UMAP; not in runner | LOW | `python -m pytest tests -q` |
| `src/run_analysis.py` | `archive/legacy_code/run_analysis.py` | 4-line stub | LOW | `python -m pytest tests -q` |
| `src/run_full_pipeline.py` | `archive/legacy_code/run_full_pipeline.py` | Superseded by `reproduce.py` | LOW | `python -m pytest tests -q` |
| `src/slerp_latent_trajectory_analysis.py` | `archive/legacy_code/slerp_latent_trajectory_analysis.py` | `sys.path` hack | LOW | `python -m pytest tests -q` |
| `src/Симуляція.py` | `archive/legacy_code/Симуляція.py` | Non-ASCII filename; writes to root | LOW | `python -m pytest tests -q` |
| `src/Симуляція_2.py` | `archive/legacy_code/Симуляція_2.py` | Non-ASCII filename | LOW | `python -m pytest tests -q` |

### Group 3: Old Experiments

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `src/experiments/analyze_beta_diversity.py` | `archive/old_experiments/analyze_beta_diversity.py` | `sys.path.insert` hack; Cyrillic import | LOW | `python -m pytest tests -q` |
| `src/experiments/backtest_vae_models.py` | `archive/old_experiments/backtest_vae_models.py` | `allow_pickle=True`; unused imports | LOW | `python -m pytest tests -q` |
| `src/experiments/inspect_orders_generated.py` | `archive/old_experiments/inspect_orders_generated.py` | `allow_pickle=True`; 2-line script | LOW | `python -m pytest tests -q` |
| `src/experiments/robustness_architecture_analysis.py` | `archive/old_experiments/robustness_architecture_analysis.py` | Legacy sweep; unused imports | LOW | `python -m pytest tests -q` |
| `scripts/experiments/aggregate_architecture_metrics.py` | `archive/old_experiments/aggregate_architecture_metrics.py` | Legacy metric aggregation | LOW | — |
| `scripts/experiments/generate_architecture_trajectory_metrics.py` | `archive/old_experiments/generate_architecture_trajectory_metrics.py` | Legacy trajectory metrics | LOW | — |
| `scripts/experiments/generate_latent_trajectories_per_arch.py` | `archive/old_experiments/generate_latent_trajectories_per_arch.py` | Legacy | LOW | — |
| `scripts/experiments/generate_principal_vectors.py` | `archive/old_experiments/generate_principal_vectors.py` | Legacy | LOW | — |

### Group 4: Legacy Results

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `results/generation_1/` | `archive/invalidated_geometry/generation_1/` | Contains invalidated Fisher `.npy` and analysis outputs | MEDIUM — verify no active code references it at runtime | `python -m pytest tests -q` |
| `results/generation_2/` | `archive/legacy_results/generation_2/` | Robustness sweep outputs; not referenced by active pipeline | MEDIUM | `python -m pytest tests -q` |
| `results/reproduction_20260618_*/` (all but latest) | `archive/legacy_results/old_repro_runs/` | Intermediate development runs | LOW | — |

### Group 5: Original Research

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `docs/Стаття_Результат_Дослідження.md` | `archive/original_research/Стаття_Результат_Дослідження.md` | Original article; 242 KB; non-ASCII filename; contains invalidated claims | LOW | — |

### Group 6: Orphan Artifacts (Root Pollution)

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `Z_train.npy` | `archive/orphan_artifacts/Z_train.npy` | Root pollution | LOW | — |
| `Z_train_enriched.npy` | `archive/orphan_artifacts/Z_train_enriched.npy` | Root pollution | LOW | — |
| `train_labels.npy` | `archive/orphan_artifacts/train_labels.npy` | Root pollution | LOW | — |
| `train_labels.csv` | `archive/orphan_artifacts/train_labels.csv` | Root pollution | LOW | — |
| `train_labels_enriched.npy` | `archive/orphan_artifacts/train_labels_enriched.npy` | Root pollution | LOW | — |
| `vae_model_weights.pth` (root) | `archive/orphan_artifacts/vae_model_weights_root_duplicate.pth` | Duplicate of `models/vae_model_weights.pth` | MEDIUM — verify hash match before removing | — |
| `generated_valid_thetas.npy` | `archive/orphan_artifacts/generated_valid_thetas.npy` | Forbidden legacy artifact from `Симуляція.py` | LOW | — |
| `orders_generated.npy` | `archive/orphan_artifacts/orders_generated.npy` | Root pollution | LOW | — |
| `z_generated.npy` | `archive/orphan_artifacts/z_generated.npy` | Root pollution | LOW | — |
| `z_generated.csv` | `archive/orphan_artifacts/z_generated.csv` | Root pollution | LOW | — |
| `z_train_ar.npy` | `archive/orphan_artifacts/z_train_ar.npy` | Root pollution | LOW | — |
| `z_train_ar.csv` | `archive/orphan_artifacts/z_train_ar.csv` | Root pollution | LOW | — |
| `z_train_garch.npy` | `archive/orphan_artifacts/z_train_garch.npy` | Root pollution | LOW | — |
| `z_train_garch.csv` | `archive/orphan_artifacts/z_train_garch.csv` | Root pollution | LOW | — |
| `z_interp_1.npy` … `5.npy` | `archive/orphan_artifacts/` | Root pollution | LOW | — |
| `interpolation_detailed_1.csv` … `5.csv` | `archive/orphan_artifacts/` | Root pollution | LOW | — |
| `latent_space_tsne.png` | `archive/orphan_artifacts/latent_space_tsne.png` | Root pollution | LOW | — |
| `latent_space_umap.png` | `archive/orphan_artifacts/latent_space_umap.png` | Root pollution | LOW | — |
| `coefficients_sums_by_lambda.png` | `archive/orphan_artifacts/coefficients_sums_by_lambda.png` | Root pollution | LOW | — |
| `norms_and_correlations.png` | `archive/orphan_artifacts/norms_and_correlations.png` | Root pollution | LOW | — |
| `validity_by_lambda.png` | `archive/orphan_artifacts/validity_by_lambda.png` | Root pollution | LOW | — |

### Group 7: Delete Candidates

| source path | reason | risk level |
|---|---|---|
| `src/visualize_arma_manifold.py` | Zero bytes | NONE |
| `src/experiments/robustness_architecture_stability_analysis.py` | Zero bytes | NONE |
| `src/experiments/robustness_seed_analysis.py` | Zero bytes | NONE |
| `src/experiments/robustness_stability_analysis.py` | Zero bytes | NONE |
| `docs/temp_edit.md` | Scratch file | NONE |
| `docs/VAE_Architecture_Technical_Specification.md` | Zero bytes | NONE |
| `docs/Стаття_Результат_Дослідження_updated.md` | Byte-identical to original; duplicate | LOW (verify before delete) |
| `data/generated/generated_valid_thetas.npy` | Forbidden artifact; same as root copy | LOW |

### Group 8: Legacy Models

| source path | target path | reason | risk level | tests required after move |
|---|---|---|---|---|
| `models/vae_beta01.pth` | `archive/legacy_models/vae_beta01.pth` | Beta sweep; not used by pipeline | LOW | `python -m pytest tests -q` |
| `models/vae_beta05.pth` | `archive/legacy_models/vae_beta05.pth` | Same | LOW | Same |
| `models/vae_beta10.pth` | `archive/legacy_models/vae_beta10.pth` | Same | LOW | Same |
| `models/vae_beta20.pth` | `archive/legacy_models/vae_beta20.pth` | Same | LOW | Same |
| `models/vae_beta50.pth` | `archive/legacy_models/vae_beta50.pth` | Same | LOW | Same |
| `models/vae_deeper.pth` | `archive/legacy_models/vae_deeper.pth` | Architecture experiment | LOW | Same |
| `models/vae_latent16.pth` | `archive/legacy_models/vae_latent16.pth` | Architecture experiment | LOW | Same |
| `models/vae_wider.pth` | `archive/legacy_models/vae_wider.pth` | Architecture experiment | LOW | Same |

---

## Required Validation After All Moves

After executing the archive moves, run:

```bash
python -m pytest tests -q
python -m src.reproduce --config configs/demo.yaml --stages 1,2,3,4,5,6,7
```

Both must pass before the cleanup patch is accepted.

---

## Archive Tracking Policy (Phase 1R Finalization)

As part of Phase 1R finalization, **Policy B (Archive binaries are local-only)** is adopted:
- Legacy binary artifacts (`.npy`, `.pth`, `.csv`, `.png`) in `archive/` are local-only and excluded from public git tracking to prevent repository size bloat.
- All code (`.py`) and documentation (`.md`) inside `archive/` are committed and tracked in public git.
- The root `.gitignore` enforces this rule.

---

## Medium-Risk Moves

The following moves have MEDIUM risk due to potential test references or
manifest citations. They should be executed last and verified separately:

1. `results/generation_1/` — verify that no test fixture uses this path.
2. `results/generation_2/` — same check.
3. Root `vae_model_weights.pth` — verify SHA-256 matches `models/vae_model_weights.pth` before removing.

---

## Files That Must NOT Be Moved

These files are SHA-256 tracked by Stage 6 and/or Stage 7 manifests:

- `docs/CLAIM_VERDICT_MATRIX.md`
- `docs/GEOMETRY_VALIDITY_STATUS.md`
- `docs/RESEARCH_INTEGRITY.md`
- `models/vae_model_weights.pth`
- `configs/demo.yaml`
- `results/reproduction_20260620_085314/` (entire latest run)

Moving any of these would break the manifest hash verification in
subsequent Stage 7 runs.
