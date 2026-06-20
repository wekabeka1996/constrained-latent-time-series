# File Classification Manifest

**Phase:** 1P  
**Date:** 2026-06-20  
**Anchor:** `results/reproduction_20260620_085314/`

Classification labels:
- `ACTIVE_CORE` — part of the canonical pipeline; must not be removed
- `ACTIVE_TEST` — active test suite
- `ACTIVE_DOC` — current documentation required for audit trail
- `ACTIVE_CONFIG` — active configuration
- `ACTIVE_MODEL_CHECKPOINT` — primary model weights used by the runner
- `ACTIVE_REPRO_OUTPUT_EXAMPLE` — example reproduction output (latest run)
- `ARCHIVE_LEGACY_CODE` — original exploratory code; not in active pipeline; retain for audit history
- `ARCHIVE_LEGACY_RESULT` — legacy experimental outputs; invalidated or unverified
- `ARCHIVE_INVALIDATED_GEOMETRY` — outputs from the buggy Fisher metric script; do not cite
- `ARCHIVE_ORIGINAL_ARTICLE` — source research article (Ukrainian); retain for record
- `DELETE_CANDIDATE` — zero-byte, scratch, or duplicate files with no value
- `REVIEW_REQUIRED` — unclear status; needs human decision before action

---

## src/ — Core Package

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `src/config.py` | `ACTIVE_CORE` | SSOT for config schema and integrity guard | NO | `pydantic`, `yaml` | Keep; fix P2 lint |
| `src/data_generator.py` | `ACTIVE_CORE` | Canonical synthetic data generator | NO | `src/vector_schema.py` | Keep; fix unused `Literal` import (P2) |
| `src/geometry.py` | `ACTIVE_CORE` | Corrected pullback Fisher metric | NO | `torch`, `numpy` | Keep |
| `src/io_utils.py` | `ACTIVE_CORE` | `require_file` fail-fast utility | NO | — | Keep |
| `src/reproduce.py` | `ACTIVE_CORE` | 7-stage reproduction runner | NO | all active core modules | Keep; fix `allow_pickle` (P0) |
| `src/vae.py` | `ACTIVE_CORE` | VAE model architecture and checkpoint loader | NO | `torch` | Keep |
| `src/validation.py` | `ACTIVE_CORE` | Structural validation and layout classifier | NO | `src/vector_schema.py` | Keep |
| `src/vector_schema.py` | `ACTIVE_CORE` | Parameter layout schema SSOT | NO | — | Keep |
| `src/train_vae_model.py` | `REVIEW_REQUIRED` | Trains the VAE; standalone; not called by runner but produces the checkpoint | NO | `src/vae.py` | Evaluate: keep as documented standalone entry point or integrate as a runner stage |

---

## src/ — Legacy Analysis Scripts (Standalone, Not in Pipeline)

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `src/compute_fisher_metric.py` | `ARCHIVE_INVALIDATED_GEOMETRY` | P0 Jacobian bug; outputs in `results/generation_1/fisher_metrics/` are invalid | YES | `results/generation_1/` hardcoded paths | Move to `archive/invalidated_geometry/` |
| `src/regularized_fisher_metric.py` | `ARCHIVE_INVALIDATED_GEOMETRY` | Depends on invalidated Fisher metrics; hardcoded Windows path in comment | YES | `src/compute_fisher_metric.py` implicitly | Move to `archive/invalidated_geometry/` |
| `src/analyze_fisher_metrics.py` | `ARCHIVE_INVALIDATED_GEOMETRY` | Loads and analyses invalidated Fisher `.npy` files; bare `except` | YES | `results/generation_1/fisher_metrics/` | Move to `archive/invalidated_geometry/` |
| `src/analyze_interpolation_results.py` | `ARCHIVE_LEGACY_CODE` | Hardcoded `results/generation_1/` paths; not in pipeline | YES | `results/generation_1/analysis/` | Move to `archive/legacy_code/` |
| `src/analyze_interpolation_norm_vs_validity.py` | `ARCHIVE_LEGACY_CODE` | Hardcoded `results/generation_1/` paths; not in pipeline | YES | `results/generation_1/analysis/` | Move to `archive/legacy_code/` |
| `src/analyze_roots_boundary_validity.py` | `ARCHIVE_LEGACY_CODE` | Hardcoded `results/generation_1/` paths; not in pipeline | YES | `results/generation_1/analysis/` | Move to `archive/legacy_code/` |
| `src/baseline_random_search.py` | `ARCHIVE_LEGACY_CODE` | Imports from `src.Симуляція_2`; not in pipeline | YES | `src/Симуляція_2.py` | Move to `archive/legacy_code/` |
| `src/generate_latent_vectors.py` | `ARCHIVE_LEGACY_CODE` | Standalone; E402 import at line 57; loads `generated_valid_thetas.npy`; not in pipeline | YES | `models/vae_model_weights.pth`, `generated_valid_thetas.npy` | Move to `archive/legacy_code/` |
| `src/generate_enriched_training_data.py` | `REVIEW_REQUIRED` | Unused `os` import; not in pipeline; generates enriched data | NO | — | Review if enriched data is needed; otherwise archive |
| `src/latent_space_interpolation_analysis.py` | `ARCHIVE_LEGACY_CODE` | Legacy interpolation analysis; unused `matplotlib` import; logic partially ported into `validation.py` | YES | `src/Симуляція_2.py` | Move to `archive/legacy_code/` |
| `src/latent_space_visualization.py` | `ARCHIVE_LEGACY_CODE` | Legacy t-SNE/UMAP visualization; not in pipeline | YES | `umap-learn`, `sklearn` | Move to `archive/legacy_code/` |
| `src/run_analysis.py` | `ARCHIVE_LEGACY_CODE` | 4-line stub calling legacy scripts | YES | legacy scripts | Move to `archive/legacy_code/` |
| `src/run_full_pipeline.py` | `ARCHIVE_LEGACY_CODE` | Legacy pipeline entry point; superseded by `src/reproduce.py` | YES | legacy scripts | Move to `archive/legacy_code/` |
| `src/slerp_latent_trajectory_analysis.py` | `ARCHIVE_LEGACY_CODE` | `sys.path.append` hack; not in pipeline | YES | `sys.path` manipulation | Move to `archive/legacy_code/` |
| `src/Симуляція.py` | `ARCHIVE_LEGACY_CODE` | Non-ASCII filename; writes `generated_valid_thetas.npy` to root; not in pipeline | YES | — | Move to `archive/legacy_code/` |
| `src/Симуляція_2.py` | `ARCHIVE_LEGACY_CODE` | Non-ASCII filename; writes/reads `generated_valid_thetas.npy`; not in pipeline | YES | — | Move to `archive/legacy_code/` |
| `src/visualize_arma_manifold.py` | `DELETE_CANDIDATE` | Zero bytes; no content | YES | — | Delete |

---

## src/experiments/ — Legacy Experiments

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `src/experiments/analyze_beta_diversity.py` | `ARCHIVE_LEGACY_CODE` | `sys.path.insert` hack; imports Cyrillic module; not in pipeline | YES | `src/Симуляція_2.py` | Move to `archive/old_experiments/` |
| `src/experiments/backtest_vae_models.py` | `ARCHIVE_LEGACY_CODE` | `allow_pickle=True`; unused `yfinance`; ambiguous `l`; not called by Stage 6 | YES | arch, statsmodels | Move to `archive/old_experiments/` |
| `src/experiments/inspect_orders_generated.py` | `ARCHIVE_LEGACY_CODE` | `allow_pickle=True`; hardcoded `orders_generated.npy`; 2-line script | YES | `orders_generated.npy` at root | Move to `archive/old_experiments/` |
| `src/experiments/robustness_architecture_analysis.py` | `ARCHIVE_LEGACY_CODE` | Legacy architecture sweep; unused `os`/`Dataset` imports | YES | — | Move to `archive/old_experiments/` |
| `src/experiments/robustness_architecture_stability_analysis.py` | `DELETE_CANDIDATE` | Zero bytes | YES | — | Delete |
| `src/experiments/robustness_seed_analysis.py` | `DELETE_CANDIDATE` | Zero bytes | YES | — | Delete |
| `src/experiments/robustness_stability_analysis.py` | `DELETE_CANDIDATE` | Zero bytes | YES | — | Delete |

---

## tests/ — Test Suite

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `tests/test_config.py` | `ACTIVE_TEST` | Tests `src/config.py` | NO | `src/config.py` | Keep |
| `tests/test_data_generator.py` | `ACTIVE_TEST` | Tests `src/data_generator.py` | NO | `src/data_generator.py` | Keep |
| `tests/test_fail_fast_integrity.py` | `ACTIVE_TEST` | Tests integrity contract enforcement | NO | `src/config.py` | Keep |
| `tests/test_geometry.py` | `ACTIVE_TEST` | Tests `src/geometry.py` | NO | `src/geometry.py` | Keep |
| `tests/test_reproduce_skeleton.py` | `ACTIVE_TEST` | Tests reproduce CLI skeleton | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage1_stage2.py` | `ACTIVE_TEST` | Tests Stages 1–2 | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage3.py` | `ACTIVE_TEST` | Tests Stage 3 | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage4.py` | `ACTIVE_TEST` | Tests Stage 4 | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage5_geometry.py` | `ACTIVE_TEST` | Tests Stage 5 | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage6_empirical.py` | `ACTIVE_TEST` | Tests Stage 6 | NO | `src/reproduce.py` | Keep |
| `tests/test_reproduce_stage7_verdict.py` | `ACTIVE_TEST` | Tests Stage 7 | NO | `src/reproduce.py` | Keep |
| `tests/test_vae.py` | `ACTIVE_TEST` | Tests `src/vae.py` | NO | `src/vae.py` | Keep |
| `tests/test_validation.py` | `ACTIVE_TEST` | Tests `src/validation.py` | NO | `src/validation.py` | Keep |
| `tests/test_vector_schema.py` | `ACTIVE_TEST` | Tests `src/vector_schema.py` | NO | `src/vector_schema.py` | Keep |

---

## docs/

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `docs/CLAIM_VERDICT_MATRIX.md` | `ACTIVE_DOC` | SHA-256 tracked by Stage 6 and 7 manifests | NO | Stages 6, 7 | Keep |
| `docs/FINDINGS_SIMPLE.md` | `ACTIVE_DOC` | Plain-language findings (Phase 1P) | NO | — | Keep |
| `docs/GEOMETRY_VALIDITY_STATUS.md` | `ACTIVE_DOC` | SHA-256 tracked by Stage 7 manifest | NO | Stage 7 | Keep |
| `docs/NUMERICAL_PRECISION.md` | `ACTIVE_DOC` | Precision policy | NO | — | Keep |
| `docs/REPRODUCTION_PLAN.md` | `ACTIVE_DOC` | Stage spec | NO | — | Keep |
| `docs/RESEARCH_INTEGRITY.md` | `ACTIVE_DOC` | SHA-256 tracked by Stage 7 manifest | NO | Stage 7 | Keep |
| `docs/WEIGHTS_GOVERNANCE.md` | `ACTIVE_DOC` | Checkpoint policy | NO | — | Keep |
| `docs/REPO_FINALIZATION_AUDIT.md` | `ACTIVE_DOC` | Phase 1P audit (new) | NO | — | Keep |
| `docs/FILE_CLASSIFICATION_MANIFEST.md` | `ACTIVE_DOC` | Phase 1P classification (new) | NO | — | Keep |
| `docs/ARCHIVE_PLAN.md` | `ACTIVE_DOC` | Phase 1P archive plan (new) | NO | — | Keep |
| `docs/README_BLUEPRINT.md` | `ACTIVE_DOC` | Phase 1P README blueprint (new) | NO | — | Keep |
| `docs/Phase_I_Summary_Report.md` | `ACTIVE_DOC` | Internal audit history | NO | — | Keep (internal) |
| `docs/INTEGRATION_GUIDE.md` | `REVIEW_REQUIRED` | May reference legacy scripts | NO | — | Review for stale references before publication |
| `docs/latent_manifold_sections.md` | `REVIEW_REQUIRED` | Contains geometry claims; must be audited against verdict matrix | NO | — | Review before publishing |
| `docs/updated_methodology.md` | `REVIEW_REQUIRED` | May reference legacy methods | NO | — | Review for invalidated claims |
| `docs/temp_edit.md` | `DELETE_CANDIDATE` | Scratch file | YES | — | Delete |
| `docs/VAE_Architecture_Technical_Specification.md` | `DELETE_CANDIDATE` | Zero bytes | YES | — | Delete |
| `docs/# Візуалізація латентного простору VAE.md` | `REVIEW_REQUIRED` | `#` in filename (shell hazard); content may be useful | NO | — | Rename to `docs/latent_space_visualization_notes.md`; review content |
| `docs/Стаття_Результат_Дослідження.md` | `ARCHIVE_ORIGINAL_ARTICLE` | Legacy research article; 242 KB; contains invalidated claims | NO | — | Move to `archive/original_research/`; SHA-256 preserved for audit |
| `docs/Стаття_Результат_Дослідження_updated.md` | `DELETE_CANDIDATE` | Byte-for-byte identical to `Стаття_Результат_Дослідження.md` | YES | — | Delete (duplicate); keep original |

---

## models/

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `models/vae_model_weights.pth` | `ACTIVE_MODEL_CHECKPOINT` | Primary checkpoint used by `configs/demo.yaml` | NO | `src/vae.py` | Keep; document hash in `WEIGHTS_GOVERNANCE.md` |
| `models/vae_beta01.pth` | `ARCHIVE_LEGACY_CODE` | Beta sensitivity sweep; not used by active pipeline | NO | — | Move to `archive/legacy_models/` or keep with clear label |
| `models/vae_beta05.pth` | `ARCHIVE_LEGACY_CODE` | Same | NO | — | Same |
| `models/vae_beta10.pth` | `ARCHIVE_LEGACY_CODE` | Same | NO | — | Same |
| `models/vae_beta20.pth` | `ARCHIVE_LEGACY_CODE` | Same | NO | — | Same |
| `models/vae_beta50.pth` | `ARCHIVE_LEGACY_CODE` | Same | NO | — | Same |
| `models/vae_deeper.pth` | `ARCHIVE_LEGACY_CODE` | Architecture experiment; not in active pipeline | NO | — | Move to `archive/legacy_models/` or keep with clear label |
| `models/vae_latent16.pth` | `ARCHIVE_LEGACY_CODE` | Latent dimension experiment | NO | — | Same |
| `models/vae_wider.pth` | `ARCHIVE_LEGACY_CODE` | Architecture experiment | NO | — | Same |

---

## configs/

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `configs/demo.yaml` | `ACTIVE_CONFIG` | Primary public demo config | NO | `src/config.py` | Keep |

---

## data/generated/ (Legacy Generated Data — Not from Runner)

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `data/generated/generated_valid_thetas.npy` | `ARCHIVE_LEGACY_RESULT` | Written by legacy `Симуляція.py`; not produced by runner | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/train_labels.npy` | `ARCHIVE_LEGACY_RESULT` | Legacy data; runner produces its own isolated copies | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/z_generated.npy` | `ARCHIVE_LEGACY_RESULT` | Legacy generated latent vectors | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/z_interp*.npy` (×6) | `ARCHIVE_LEGACY_RESULT` | Legacy interpolation artifacts | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/Z_train.npy` | `ARCHIVE_LEGACY_RESULT` | Legacy training data | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/z_train_ar.npy` | `ARCHIVE_LEGACY_RESULT` | Legacy training data | YES | — | Move to `archive/orphan_artifacts/` |
| `data/generated/z_train_garch.npy` | `ARCHIVE_LEGACY_RESULT` | Legacy training data | YES | — | Move to `archive/orphan_artifacts/` |

---

## results/

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `results/generation_1/` | `ARCHIVE_INVALIDATED_GEOMETRY` | Contains invalidated Fisher `.npy` files and legacy analysis outputs | NO | `CLAIM_VERDICT_MATRIX.md` references it for audit history | Do NOT delete; move to `archive/invalidated_geometry/` when archiving |
| `results/generation_2/` | `ARCHIVE_LEGACY_RESULT` | Contains robustness sweep results; geometry from same buggy script | NO | `CLAIM_VERDICT_MATRIX.md` references `robustness/` for CLAIM_06 | Do NOT delete; move to `archive/legacy_results/` when archiving |
| `results/reproduction_20260620_085314/` | `ACTIVE_REPRO_OUTPUT_EXAMPLE` | Latest full Stage 1–7 run; manifest and reports are the evidence anchor | NO | Stage 7 manifest hashes | Keep; document as canonical latest run |
| `results/reproduction_20260618_*/` (earlier runs) | `ARCHIVE_LEGACY_RESULT` | Intermediate development runs | YES (after confirming latest run is complete) | — | Move to `archive/old_repro_runs/` |

---

## scripts/experiments/

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `scripts/experiments/aggregate_architecture_metrics.py` | `ARCHIVE_LEGACY_CODE` | Legacy architecture metric aggregation | YES | `results/generation_2/` | Move to `archive/old_experiments/` |
| `scripts/experiments/generate_architecture_trajectory_metrics.py` | `ARCHIVE_LEGACY_CODE` | Same | YES | — | Move to `archive/old_experiments/` |
| `scripts/experiments/generate_latent_trajectories_per_arch.py` | `ARCHIVE_LEGACY_CODE` | Same | YES | — | Move to `archive/old_experiments/` |
| `scripts/experiments/generate_principal_vectors.py` | `ARCHIVE_LEGACY_CODE` | Same | YES | — | Move to `archive/old_experiments/` |

---

## Root-Level Files

| path | classification | reason | safe_to_move_now | depends_on | recommended_action |
|---|---|---|---|---|---|
| `README.md` | `REVIEW_REQUIRED` | Stale Ukrainian content with invalidated claims; needs replacement | NO | — | Replace with content from `docs/README_BLUEPRINT.md` after approval |
| `ROADMAP.md` | `REVIEW_REQUIRED` | May reference future work; review for stale claims | NO | — | Review before publishing |
| `requirements.txt` | `ACTIVE_CORE` | Dependency specification | NO | — | Keep |
| `Z_train.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy data | YES | — | Move to `archive/orphan_artifacts/` |
| `Z_train_enriched.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `train_labels.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `train_labels.csv` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `train_labels_enriched.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `vae_model_weights.pth` | `ARCHIVE_LEGACY_RESULT` | Duplicate of `models/vae_model_weights.pth`; root pollution | YES | — | Verify both files are identical, then remove root copy |
| `generated_valid_thetas.npy` | `ARCHIVE_LEGACY_RESULT` | Forbidden artifact from `Симуляція.py`; MUST NOT be published | YES | — | Move to `archive/orphan_artifacts/` or delete if identical copy exists elsewhere |
| `orders_generated.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution; source unclear | YES | — | Move to `archive/orphan_artifacts/` |
| `z_generated.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_generated.csv` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_train_ar.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_train_ar.csv` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_train_garch.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_train_garch.csv` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `z_interp_1.npy` … `z_interp_5.npy` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `interpolation_detailed_1.csv` … `5.csv` | `ARCHIVE_LEGACY_RESULT` | Root pollution | YES | — | Move to `archive/orphan_artifacts/` |
| `latent_space_tsne.png` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy plot | YES | — | Move to `archive/orphan_artifacts/` |
| `latent_space_umap.png` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy plot | YES | — | Move to `archive/orphan_artifacts/` |
| `coefficients_sums_by_lambda.png` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy plot | YES | — | Move to `archive/orphan_artifacts/` |
| `norms_and_correlations.png` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy plot | YES | — | Move to `archive/orphan_artifacts/` |
| `validity_by_lambda.png` | `ARCHIVE_LEGACY_RESULT` | Root pollution; legacy plot | YES | — | Move to `archive/orphan_artifacts/` |

---

## Classification Count Summary

| Classification | Count |
|---|---|
| `ACTIVE_CORE` | 9 |
| `ACTIVE_TEST` | 14 |
| `ACTIVE_DOC` | 14 |
| `ACTIVE_CONFIG` | 1 |
| `ACTIVE_MODEL_CHECKPOINT` | 1 |
| `ACTIVE_REPRO_OUTPUT_EXAMPLE` | 1 |
| `ARCHIVE_LEGACY_CODE` | 22 |
| `ARCHIVE_LEGACY_RESULT` | 29 |
| `ARCHIVE_INVALIDATED_GEOMETRY` | 4 |
| `ARCHIVE_ORIGINAL_ARTICLE` | 1 |
| `DELETE_CANDIDATE` | 8 |
| `REVIEW_REQUIRED` | 9 |
| **Total** | **113** |
