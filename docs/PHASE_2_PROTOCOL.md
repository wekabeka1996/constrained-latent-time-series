# PHASE 2 PROTOCOL
# Top-Level Research Protocol Document

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval before implementation begins

---

## 1. Research Question

> **Can a constrained / rule-aware / compositional latent generative model trained on two
> structurally different time-series families A and B generate a valid third structural type C,
> where C is not simple noise, not a numeric blend, not memorization, but a mathematically valid
> composition of properties from A and B?**

This question is the fixed research framing for all Phase 2 experiments. It may not be changed
post-hoc to accommodate experimental results.

---

## 2. Hypotheses

### H0 — Null Hypothesis
Unconstrained continuous latent generative models trained only on A and B cannot reliably generate
valid unseen C without structural inductive bias. The unconstrained VAE baseline is expected to fail
because it has no mechanism to enforce ARMA stationarity, GARCH non-negativity/persistence, or
disjoint family indicator activation simultaneously.

### H1 — Alternative Hypothesis
Rule-aware / constrained / conditional / grammar-aware / compositional generative models can generate
valid C if the A+B→C composition rule is represented in architecture, grammar, loss, projection,
or evaluation protocol.

### H2 — Anti-Interpolation Hypothesis
Pure latent interpolation between A and B endpoints may produce smooth numeric transitions, but
structural validity of C requires explicit constraints or learned compositional factors.
Numeric smoothness is not structural validity. This hypothesis must be separately verified against
any latent interpolation claim.

### Hypothesis falsification criteria
- H0 is falsified if a constrained model achieves `generated_C_valid_rate ≥ PROPOSED_PROTOCOL_THRESHOLD`
  under the zero-shot protocol and this result is stable across seeds.
- H1 is supported if constrained models outperform unconstrained baselines on `generated_C_valid_rate`
  and `generated_C_composition_score`.
- H2 is tested by comparing interpolated latents decoded directly against projected/constrained decoded
  latents; both must be evaluated against the same validity criteria.

**IMPORTANT:** The numeric thresholds are TO_BE_DEFINED. They must be committed to the metric
contract (`docs/PHASE_2_METRIC_CONTRACT.md`) and marked stable before any model is trained.

---

## 3. Family Definitions

| Symbol | Name | Mathematical structure |
|--------|------|----------------------|
| A | AR / ARMA | Mean dynamics only — AR polynomial roots inside unit circle, MA polynomial roots outside unit circle |
| B | GARCH | Volatility dynamics only — ω > 0, αᵢ ≥ 0, βⱼ ≥ 0, Σα + Σβ < 1 |
| C | ARMA-GARCH | Combined mean + volatility hybrid — simultaneously valid ARMA (stationarity, invertibility) and valid GARCH (non-negativity, persistence < 1) |

Families are **mutually exclusive at the structural intent level for training**:
- Training data for zero-shot protocol: A ∪ B only — C is completely absent
- C appears only in the zero-shot evaluation holdout
- See `docs/PHASE_2_SPLIT_CONTRACT.md` for the formal split contract

### What C is NOT
- C is NOT a numeric blend of A and B parameter vectors
- C is NOT any vector where both ARMA and GARCH indicators are weakly active (threshold ambiguity)
- C is NOT an interpolated latent midpoint
- C is NOT an ARMA-only model (ARMA is a subfamily of A)
- C is NOT noise
- C is NOT an undocumented model structure

---

## 4. Zero-Shot / Few-Shot / Supervised Definitions

### 4.1 Zero-Shot C
- Training data: A ∪ B only — no C sample, no C label, no C ModelSpec, no ARMA-GARCH hybrid
- C appears only at evaluation time
- The model has never seen a complete ARMA-GARCH parameter vector during training
- A zero-shot C claim requires: zero-shot split verification (split manifest hash), `C_count_in_zero_shot_train == 0`, generated_C_valid_rate above precommitted threshold, novelty and composition scores above precommitted thresholds, seed stability

### 4.2 Few-Shot C (1%)
- Training data: A ∪ B ∪ (1% C by sample count), using an explicit seed list and explicit sample ID manifest
- The 1% C samples must be drawn from a separate pool that is held out from the zero-shot evaluation C-holdout
- Full split manifest required; manifest hash committed before training

### 4.3 Few-Shot C (5%)
- Same as 1% but with 5% C in training
- Separate seed list and sample ID manifest required
- Must be evaluated on the same C-holdout as zero-shot (those samples remain withheld)

### 4.4 Supervised Upper Bound
- Training data: A ∪ B ∪ C (all C included)
- Establishes the theoretical ceiling for C validity rate under full supervision
- This is a reference experiment, not a claim about zero-shot capability

---

## 5. Strict No-Result-Chasing Policy

The following are unconditionally prohibited. See `docs/PHASE_2_NO_RESULT_CHASING_RULES.md` for the
full catalogue:

1. Metrics must be defined and frozen **before** any training begins.
2. Thresholds must be committed to `docs/PHASE_2_METRIC_CONTRACT.md` **before** any training begins.
3. Family definitions must be frozen **before** any data generation begins.
4. Split contract must be frozen **before** any data generation begins.
5. No post-hoc metric changes.
6. No post-hoc threshold changes.
7. No deleting failed runs.
8. No overwriting artifacts.
9. No claim without evidence.
10. Negative results must be preserved. A failed experiment is a valid result.

---

## 6. Strict No-Defaults Policy

Every research-affecting value must be explicit in the Phase 2 protocol and config.
No Pydantic default, argparse default, function-level default, or dataclass default may silently
supply a research-affecting value.

If a value is absent from the config, the system must FAIL_FAST. See `docs/PHASE_2_CONFIG_DISCIPLINE.md`.

Legacy values (seed=42, threshold=0.5, epsilon=0.05, latent_dim=8, n_arma=100, n_garch=100) are
documented as **Phase 1 legacy baseline values only**. They are not Phase 2 defaults.

---

## 7. Strict No-Training-Before-Protocol Policy

Training is blocked until ALL of the following gates are cleared:

### Gate G1 — Schema Frozen
- `docs/PHASE_2_SCHEMA_SPEC.md` must be reviewed and marked APPROVED
- The Phase 2 ModelSpec must be formally defined (not implemented — defined)
- No v[10] collision may exist in the target schema

### Gate G2 — Split Contract Frozen
- `docs/PHASE_2_SPLIT_CONTRACT.md` must be reviewed and marked APPROVED
- Family definitions (A/B/C) must be exact
- Zero-shot C exclusion rule must be explicit

### Gate G3 — Metric Contract Frozen
- `docs/PHASE_2_METRIC_CONTRACT.md` must be reviewed and marked APPROVED
- All TO_BE_DEFINED thresholds must be replaced with APPROVED_PROTOCOL_THRESHOLD values
- No TO_BE_DEFINED items may remain in the metric contract when training begins

### Gate G4 — Config Discipline Implemented
- Phase 2 config file (to be created in a future implementation task) must pass all rules in `docs/PHASE_2_CONFIG_DISCIPLINE.md`
- All silent Pydantic geometry defaults must be explicit

### Gate G5 — Environment Frozen
- `pip freeze` output committed as `requirements-freeze.txt` or equivalent lock file
- Python version documented
- torch version documented and verified functional

### Gate G6 — Test Baseline Established
- `python -m pytest tests/ -q` must pass with 0 failures (the 4 known skips are acceptable)
- All 9 torch-dependent test files must be able to collect (requires torch installed)

### Gate G7 — P1 Blockers Resolved
- See §10 (Current P1 Blockers Inherited)

### Gate G8 — Reviewer Approval
- This protocol document must be reviewed by the project owner before implementation begins
- Approval must be documented (e.g. PR approval or written acknowledgment with date and commit hash)

---

## 8. Required Gates Before Final Claim

Before any Phase 2 result is published or cited:

1. Full artifact archive must exist (`docs/PHASE_2_ARTIFACT_CONTRACT.md`)
2. All metrics must have been frozen before training (documented by metric contract hash in manifest)
3. Unconstrained VAE baseline must be compared against constrained model
4. Results must be stable across ≥ 3 independent random seeds (TO_BE_DEFINED: minimum seed count)
5. Claim language must comply with `docs/PHASE_2_NO_RESULT_CHASING_RULES.md` §11
6. Negative results must be included in the final report

---

## 9. Current Blockers Inherited from P1

These blockers were documented in `reports/PHASE_2_P1_REPO_INVENTORY.md` (v2) and
`reports/PHASE_2_P1_REVIEW_FIX_NOTES.md`. They are carried forward as P2 blockers:

| Blocker | ID | Impact |
|---------|----|--------|
| `torch` not installed in local environment | B1 | Blocks runtime VAE claims, full test collection, Gates G5/G6 |
| No `pip freeze` / lock file committed | B3 | Blocks Gate G5 |
| `code_git_commit: None` hardcoded in `build_initial_manifest()` | B2 | Blocks artifact provenance |
| v[10] collision in legacy 40D schema | S1 | Blocks C-family schema definition; Gate G1 unresolved |
| Validation thresholds are function-level defaults (not config-driven) | D1 | Blocks Gate G4 |

**None of B1, B2, B3, S1, D1 are resolved by this documentation task.**
This documentation task only freezes the protocol. Resolving blockers requires a separate implementation task (Phase 2 P3 — Implementation).

---

## 10. Sub-Document Index

| Document | Purpose | Status |
|----------|---------|--------|
| `docs/PHASE_2_PROTOCOL.md` | This document — top-level protocol | DRAFT |
| `docs/PHASE_2_RESEARCH_PROTOCOL.md` | Problem framing, scope, experiments, baselines, claim language | DRAFT |
| `docs/PHASE_2_SCHEMA_SPEC.md` | Vector schema v2 intent — no code | DRAFT |
| `docs/PHASE_2_SPLIT_CONTRACT.md` | A/B/C split rules, zero-shot/few-shot, leakage guard | DRAFT |
| `docs/PHASE_2_METRIC_CONTRACT.md` | All metrics, validity definitions, thresholds, success/failure criteria | DRAFT |
| `docs/PHASE_2_CONFIG_DISCIPLINE.md` | No-defaults policy, required explicit values, fail-fast rules | DRAFT |
| `docs/PHASE_2_ARTIFACT_CONTRACT.md` | Required artifact structure, manifest requirements | DRAFT |
| `docs/PHASE_2_NO_RESULT_CHASING_RULES.md` | Prohibition list, claim language rules | DRAFT |
| `reports/PHASE_2_P2_PROTOCOL_DESIGN_REPORT.md` | Task completion report | DRAFT |

All eight supporting documents are created in this task. All are DRAFT status pending reviewer approval.
None of them constitute implementation authorization.

---

## 11. What This Document Is NOT

- This document does NOT authorize implementation of any Phase 2 code.
- This document does NOT constitute a model training plan.
- This document does NOT claim that any constrained model works.
- This document does NOT authorize creation of `configs/phase2.yaml`.
- This document does NOT authorize creation of `src/vector_schema_v2.py`, `src/data_generator_v2.py`, `src/vae_constrained.py`, `src/train_vae_v2.py`, or any other Phase 2 source file.

Implementation is gated on reviewer approval of this protocol and resolution of all blockers.
