# Phase 2 P22: Artifact-Backed Baseline Evidence Smoke Report

## Verdict
**PASS** - P22 successfully implemented and executed a read-only artifact-backed baseline evidence smoke run using previously generated P14 artifacts. 

## Process Evidence

**1. P14 Artifact Audit Validation:**
The P22 smoke test script validated the P14 artifacts using the P15 `audit_phase2_artifact_manifest` utility. 
Artifact Root used: `phase2_artifacts/p14_smoke_dry_run`
Audit verified: `true`

**2. Artifact Reference Selection:**
P22 successfully extracted 2 `ModelSpec` candidates from `zero_shot_train/samples.jsonl`:
- `ARMA` mean source (`sample_id: ARMA_zero_shot_train_000001_g12001_s1012001_29dcbe9e1c07`, Line: 1001)
- `GARCH` volatility source (`sample_id: GARCH_zero_shot_train_000001_g12002_s1012002_6d653cb4f1b3`, Line: 2001)

**3. Baseline Generation and Evaluation Summaries:**
Evaluated 3 baselines (6 candidates each):
- **`copy_reference`:**
  - Candidates: 3 ARMA copies, 3 GARCH copies
  - Validity Pass Rate: 100%
  - Composition Pass Rate: 0%
  - Distribution Distance (MMD-RBF): 0.0

- **`random_valid`:**
  - Candidates: 2 AR, 2 GARCH, 2 ARMA_GARCH generated dynamically with independent seeds
  - Validity Pass Rate: 100%
  - Composition Pass Rate: ~33% (matches expected for ARMA_GARCH subset)
  - Distribution Distance (MMD-RBF): ~0.467

- **`structural_composition_oracle`:**
  - Candidates: 6 ARMA_GARCH composed deterministically from ARMA mean and GARCH volatility references.
  - Validity Pass Rate: 100%
  - Composition Pass Rate: 100%
  - Distribution Distance (MMD-RBF): ~1.498

**4. Constraint Verification:**
- `no_artifact_generation`: `true`
- `no_model_training`: `true`
- `no_final_comparison`: `true`
- `no_scientific_conclusion`: `true`
- `metric_bundle_bridge_verified`: `true` (all metrics fit cleanly into `MetricBundle`)

## Status
P22 execution and strict adherence to Phase 2 primitives verified. The testing pipeline is strictly read-only and respects artifact-backed models.

**Branch**: `phase2/p22-artifact-backed-baseline-evidence-smoke`
