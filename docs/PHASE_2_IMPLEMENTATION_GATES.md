# docs/PHASE_2_IMPLEMENTATION_GATES.md — Implementation Gates Specification

## 1. Gate Overview
This document specifies the validation gates that must be cleared before any Phase 2 implementation code, datasets, or experiments can be created. The current task (P4: Governance) does **not** authorize implementation. It defines the validation matrix future agents must satisfy.

## 2. Gate G0 — Governance Present
- **Requirements:**
  - `AGENTS.md` exists and is frozen in the repository root.
  - `docs/PHASE_2_AGENT_RUNBOOK.md` exists.
  - `docs/PHASE_2_REVIEW_CHECKLIST.md` exists.
  - `docs/PHASE_2_PROMPT_SEQUENCE.md` exists.
  - The Phase 2 P4 report is committed and pushed.

## 3. Gate G1 — Schema Spec Approved
- **Requirements:**
  - `docs/PHASE_2_SCHEMA_SPEC.md` is reviewed and marked APPROVED by the project owner.
  - Max order parameters (`max_p = 5`, `max_q = 5`, `max_r = 2`, `max_s = 2`) and minimum serialization dimensions (`schema_v2_min_flat_dim = 32`) are frozen.
  - The logical structure of `ModelSpec` is frozen.
  - Representation rules ensuring non-overlapping slots and no family-conditional index interpretation are locked.
  - Implementation form (dataclass vs. Pydantic class) remains deferred.

## 4. Gate G2 — Split Contract Approved
- **Requirements:**
  - `docs/PHASE_2_SPLIT_CONTRACT.md` is reviewed and marked APPROVED.
  - Mathematical definitions of A-family, B-family, and C-family boundaries are locked.
  - Training, holdout, and evaluation sample counts for Smoke, Dev, Main, and Large benchmarks are frozen.
  - Full seed lists (`seed_generation_A`, etc.) are frozen.
  - Zero-shot C-exclusion rule (`C_count_in_zero_shot_train == 0`) is explicitly enforced.
  - Few-shot rounding and superset rules are defined and frozen.

## 5. Gate G3 — Metric Contract Approved
- **Requirements:**
  - `docs/PHASE_2_METRIC_CONTRACT.md` is reviewed and marked APPROVED.
  - Mathematical validity metrics, thresholds, composition rules, novelty scores, and simulation constraints are frozen.
  - No post-hoc modification of metrics or thresholds is allowed once a run has started.

## 6. Gate G4 — Config Discipline Approved
- **Requirements:**
  - Config key schema matches `docs/PHASE_2_CONFIG_DISCIPLINE.md`.
  - All research-affecting config keys are required (no defaults).
  - Key checking and dirty tree check are in place.
  - Approved tolerances (`validation.tolerance = 1e-8`, etc.) are locked.
  - Explicit required integrity settings are committed.

## 7. Gate G5 — Environment Ready
- **Requirements:**
  - `torch` is verified as installed and functional in the agent environment.
  - The full test suite can collect (all 9 torch-dependent test files can be loaded by pytest).
  - Python version, platform, CUDA availability, and Torch version are documented in `environment_freeze.txt`.
  - `pip freeze` lock file `requirements-phase2-freeze.txt` is generated and committed.

## 8. Gate G6 — Artifact Discipline Ready
- **Requirements:**
  - Snapshots of configurations, metric contract, and split contract are generated and hashed in the run manifest.
  - The `run_id` matches the approved format (`phase2_<stage>_<protocol>_<YYYYMMDD_HHMMSS>_<gitshort>_<seedgroup>`).
  - Mechanisms to write `git_commit.txt` and `dirty_tree_status.txt` are ready.

## 9. Gate G7 — No-Go Conditions Cleared
- **Requirements:**
  - There are no unresolved blockers from P1, P2, or P3 that affect the current task.
  - The git status shows a clean working tree.
  - All approved protocol values and seeds are present in the configuration.
  - No branch commit mismatches or missing branch evidence.

## 10. Gate G8 — Owner Review Complete
- **Requirements:**
  - The project owner has explicitly reviewed the task branch, evidence report, and diff output, and has marked the gate as ACCEPTED.

---

## 11. Gate Matrix

The following matrix maps task types to their required validation gates:

| Task Type | Required Gates | Blocked / Allowed Actions |
| :--- | :--- | :--- |
| **Documentation Only** | G0, G7, G8 | Only text/markdown changes in `docs/` or `reports/`. |
| **Schema Implementation** | G0, G1, G4, G7, G8 | Creating/writing skeleton `src/vector_schema_v2.py` code. |
| **Generator Implementation** | G0, G1, G2, G4, G7, G8 | Modifying/creating data generation pipelines. |
| **Dataset Generation** | G0, G1, G2, G4, G5, G7, G8 | Executing data generation scripts to create binary files. |
| **Metric Implementation** | G0, G1, G3, G4, G7, G8 | Coding metric formulas and validator checks. |
| **Model Implementation** | G0, G1, G4, G5, G7, G8 | Coding model architectures (`src/vae_constrained.py`). |
| **Model Training** | G0–G8 (All Gates) | Running training pipelines or updating weights. |
| **Final Claim Report** | G0–G8 (All Gates) | Authorizing success claims or generalizing conclusions. |
