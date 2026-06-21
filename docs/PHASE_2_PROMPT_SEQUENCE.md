# docs/PHASE_2_PROMPT_SEQUENCE.md — Phase 2 Prompt Sequence Plan

This document outlines the planned sequential task prompts for Phase 2 implementation. **No future prompt is authorized by this document.** Each prompt step must be explicitly initiated and approved by the user.

## 1. Completed Prompts
- **PHASE_2_P1_REPO_INVENTORY_AND_FAILURE_MAP:** Completed (P1 inventory, failure map, next steps, review fix notes).
- **PHASE_2_P2_PROTOCOL_AND_SCHEMA_DESIGN:** Completed (P2 protocol, schema specification, metric contract, config discipline, split contract, design report).
- **PHASE_2_P3_RESOLVE_PROTOCOL_CONSTANTS_AND_APPROVAL_VALUES:** Completed (Resolved constants, thresholds, and seeds; committed report).
- **PHASE_2_P4_AGENT_GOVERNANCE_AND_RUNBOOK:** Active/completed by this task (Creates AGENTS.md, runbook, gates list, checklists, usage rules, sequence, and report).

## 2. Future Prompt Sequence (Not Yet Authorized)

### PHASE_2_P5 — Schema Guard Skeleton & ModelSpec Implementation Plan
- **Purpose:** Prepare for schema v2 implementation. Build class skeletons and detail the exact implementation language choice (e.g. standard dataclass or Pydantic class) and validate it conceptually.
- **Allowed Files:** `docs/PHASE_2_MODELSPEC_IMPLEMENTATION_PLAN.md`, `reports/PHASE_2_P5_SKELETON_REPORT.md`, `src/vector_schema_v2.py` (Skeleton class structure only, no implementation).
- **Forbidden Files:** All other source files under `src/` (no functional code changes), all files in `tests/`, `configs/`, `models/`, or `archive/`.
- **Required Gates:** G0 (Governance), G1 (Schema Spec Approved), G7 (No-Go Cleared), G8 (Owner Review).
- **Expected Report:** P5 Skeleton Report containing branch/commit evidence, ModelSpec implementation plan choice, skeleton files list, and final verdict.

### PHASE_2_P6 — Schema v2 Implementation & Validation Validators
- **Purpose:** Code the functional implementation of the schema v2 encoder, decoder, companion companions, and validator utilities.
- **Allowed Files:** `src/vector_schema_v2.py`, `tests/test_phase2_schema.py`, `reports/PHASE_2_P6_SCHEMA_REPORT.md`.
- **Forbidden Files:** All other `src/` files (generators, models, trainers), `configs/`, `models/`.
- **Required Gates:** G0, G1, G4, G7, G8.
- **Expected Report:** P6 Schema Report detailing implementation details, unit tests execution log, and final verdict.

### PHASE_2_P7 — Generator v2 & Parameter Constraint Guards
- **Purpose:** Code the Phase 2 data generator. Clean up Phase 1 GARCH beta bug and enforce GARCH positivity and stationarity constraints at sample generation time.
- **Allowed Files:** `src/data_generator_v2.py`, `tests/test_phase2_generator.py`, `reports/PHASE_2_P7_GENERATOR_REPORT.md`.
- **Forbidden Files:** Models, trainers, configs, `src/vae_constrained.py`.
- **Required Gates:** G0, G1, G2, G4, G7, G8.
- **Expected Report:** P7 Generator Report detailing generator math, constraint checks, tests results, and final verdict.

### PHASE_2_P8 — Dataset Builder & Zero-Shot Split Guards
- **Purpose:** Write the pipeline script to assemble datasets and manifests. Enforce split guards and verification hashes.
- **Allowed Files:** `src/phase2/dataset_builder.py`, `tests/test_phase2_split.py`, `reports/PHASE_2_P8_DATASET_REPORT.md`.
- **Forbidden Files:** Model training, model definitions.
- **Required Gates:** G0, G1, G2, G4, G5, G7, G8.
- **Expected Report:** P8 Dataset Report, showing that `C_count_in_zero_shot_train == 0` exactly.

### PHASE_2_P9 — Metric Contract Implementation
- **Purpose:** Code the validation checks, novelty distance estimators, composition scores, MMD distribution distance, and simulation-based statistical tests.
- **Allowed Files:** `src/phase2/metrics.py`, `tests/test_phase2_metrics.py`, `reports/PHASE_2_P9_METRICS_REPORT.md`.
- **Forbidden Files:** Training pipelines, models.
- **Required Gates:** G0, G1, G3, G4, G7, G8.
- **Expected Report:** P9 Metrics Report, verifying metric formulas on mock vectors.

### PHASE_2_P10 — Legacy Baseline Reproduction under v2 Metrics
- **Purpose:** Run baseline evaluation of the legacy Phase 1 unconstrained VAE under the new Phase 2 schema and metrics. Establish the H0 baseline.
- **Allowed Files:** `src/phase2/eval_baseline.py`, `reports/PHASE_2_P10_BASELINE_REPORT.md`.
- **Forbidden Files:** Constrained model architectures, training.
- **Required Gates:** G0–G6, G7, G8.
- **Expected Report:** P10 Baseline Report showing that legacy VAE achieves ~0% reconstruction validity on C.

### PHASE_2_P11 — beta-VAE & Conditional VAE (cVAE) Implementations
- **Purpose:** Implement B2 (beta-VAE) and B3 (cVAE) baselines.
- **Allowed Files:** `src/vae_constrained.py` (Base classes / conditional structure), `src/phase2/train_baselines.py`, `tests/test_baselines.py`, `reports/PHASE_2_P11_BASELINES_REPORT.md`.
- **Forbidden Files:** Composer, Experts, Grammar decoders.
- **Required Gates:** G0–G8 (All Gates).
- **Expected Report:** P11 Baselines Report detailing models, loss weights, training outcomes.

### PHASE_2_P12 — Gumbel Softmax Categorical-Structure VAE
- **Purpose:** Implement B4 (Gumbel-Softmax VAE).
- **Allowed Files:** `src/vae_constrained.py`, `src/phase2/train_gumbel.py`, `reports/PHASE_2_P12_GUMBEL_REPORT.md`.
- **Forbidden Files:** Grammar, MoE.
- **Required Gates:** G0–G8.
- **Expected Report:** P12 Gumbel VAE Report.

### PHASE_2_P13 — Grammar / Schema Decoder
- **Purpose:** Implement B5 (Grammar projection layer).
- **Allowed Files:** `src/vae_constrained.py`, `src/phase2/train_grammar.py`, `reports/PHASE_2_P13_GRAMMAR_REPORT.md`.
- **Forbidden Files:** Mixture-of-Experts.
- **Required Gates:** G0–G8.
- **Expected Report:** P13 Grammar Decoder Report.

### PHASE_2_P14 — Mixture-of-Experts (MoE) Composer
- **Purpose:** Implement B6 (MoE Composer).
- **Allowed Files:** `src/vae_constrained.py`, `src/phase2/train_moe.py`, `reports/PHASE_2_P14_MOE_REPORT.md`.
- **Required Gates:** G0–G8.
- **Expected Report:** P14 MoE Composer Report.

### PHASE_2_P15 — Unified Benchmarking Runner
- **Purpose:** Implement unified benchmarking runner to compare all baseline and constrained models under zero-shot, 1% few-shot, and 5% few-shot conditions.
- **Allowed Files:** `src/phase2/run_benchmark.py`, `reports/PHASE_2_P15_BENCHMARK_REPORT.md`.
- **Required Gates:** G0–G8.
- **Expected Report:** P15 Benchmark Report compiling all validity, composition, and novelty scores.

### PHASE_2_P16 — Independent Audit & Publication Guard
- **Purpose:** Perform final independent audit of manifests, contract hashes, and git tree states. Sign off on claims.
- **Allowed Files:** `reports/PHASE_2_P16_AUDIT_REPORT.md`.
- **Forbidden Files:** Model modification, dataset changes.
- **Required Gates:** G0–G8 (All Gates).
- **Expected Report:** P16 Audit Report verifying split compliance, locking final verdict accepting/rejecting the composition claim.
