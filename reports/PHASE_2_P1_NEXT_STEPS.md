# PHASE_2_P1_NEXT_STEPS.md
# Phase 2 P1: Next Steps

**Report version:** 2 (post-review fix)
**Date:** 2026-06-20
**Branch:** phase2/p1-repo-inventory
**Commit:** 67f06e142bc0cab191ef22067f84e74e3c8755fe
**Scope:** Documentation and protocol definition only — P1 is still active.

---

> [!IMPORTANT]
> **P1 is NOT complete.** The current task scope is documentation and evidence gathering only.
> Implementation of any Phase 2 model, generator, schema, metric, dataset, or training code
> is strictly **NOT authorized** under P1 or this prompt.
>
> Items in this document describe what must be done in Phase 2 protocol design (a separate task).
> They are not prescriptions or code blueprints — they are documentation of required preconditions.

---

## Summary of P1 Findings

Phase 1 of this research (completed before P1 documentation) established the following:

| Item | Outcome |
|---|---|
| Phase 1 VAE training | Completed — checkpoint `models/vae_beta50.pth` in repository |
| Phase 1 encoder clustering | PARTIALLY_CONFIRMED |
| Phase 1 decoder structural validity | INVALIDATED (0/200 = 0.0%, all LEGACY_LAYOUT_COLLISION) |
| Phase 1 interpolation validity | INVALIDATED |
| Legacy Fisher/geodesic geometry | INVALIDATED (P0 Jacobian bug — archived) |
| Corrected geometry | IMPLEMENTED in `src/geometry.py` — runtime not verified here (torch absent) |

**P1 documentation establishes the evidence baseline and preconditions for Phase 2 protocol design.**

---

## What P1 Documentation Has Produced

1. **PHASE_2_P1_REPO_INVENTORY.md** (this session, v2): Full repository inventory with git evidence, test evidence, command outputs, config audit, defaults audit, model inventory, generator inventory.

2. **PHASE_2_P1_FAILURE_MAP.md** (this session, v2): Forensic failure map with corrected C-leakage framing, hardcoded default audit with exact file:line references, known limitations clearly marked MISSING_EVIDENCE, full structural validity failure documentation.

3. **PHASE_2_P1_REVIEW_FIX_NOTES.md** (new, this session): Reviewer-identified problems, each with a specific fix applied.

---

## Known Blockers Before Phase 2 Protocol Can Begin

The following are **requirements**, not design choices. They must be resolved before Phase 2 protocol is designed.

### B1 — torch Environment (CRITICAL)

**Problem:** `torch` is not installed in the local environment. 9/14 test files cannot be collected or run.

**Required action:** Install torch. Then run:
```
python -m pytest tests/ -q
```
Expected: 200 passed, 4 skipped (per README).

**Nothing in Phase 2 that claims runtime VAE behavior can proceed without this.**

### B2 — `code_git_commit` Auto-Capture

**Problem:** `build_initial_manifest()` hardcodes `code_git_commit: None`.

**Required action:** Before Phase 2 protocol is finalized, document the exact mechanism that will be used to auto-capture the git commit hash at runtime. This could be `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` in `build_initial_manifest()`, but the implementation must be in a Phase 2 implementation task — not this P1 task.

### B3 — pip lock file

**Problem:** `requirements.txt` has minimum version bounds, not exact installed versions.

**Required action:** Capture `pip freeze` output and commit a lock file. Must be done before any Phase 2 reproducibility claim.

---

## Preconditions for Phase 2 Protocol Design (Documentation Only)

These are not code prescriptions. They describe what a Phase 2 protocol document must address.

### P1 — Schema v2 definition

The current `src/vector_schema.py` has an undefined joint layout for ARMA-GARCH vectors at `v[10]`. A Phase 2 protocol must define a vector schema that can represent all three families (A, B, C) without layout collision. This schema must be committed as a document before any code is written.

### P2 — Family split contract

A Phase 2 protocol must specify:
- Which models belong to A (AR/ARMA — mean dynamics only)
- Which models belong to B (GARCH — volatility dynamics only)
- Which models belong to C (ARMA-GARCH — combined mean + volatility)
- That C is **completely absent from training data** (zero-shot target)
- How the split will be verified at data generation time

This must be a written document with a mathematical definition, committed before any data generation script is written.

### P3 — Metric contract

A Phase 2 protocol must specify:
- What it means for a C-family sample to be "valid" (mathematical definition)
- What threshold will constitute "success" (e.g., 50% C-validity rate)
- What "zero-shot" means in this context (C never in training, C attempted only at evaluation)
- That these thresholds are committed before any model is trained

This must be a written document with numerical thresholds, committed before any training script is written.

### P4 — Config discipline specification

A Phase 2 protocol must specify:
- That every research-affecting value is explicit in the Phase 2 config
- That no Pydantic silent default is acceptable for research-affecting values
- That the geometry block is always required
- That `allow_cached_artifacts: false` is the Phase 2 default
- That `allow_random_fallbacks: false` is required

These are documentation requirements — no code changes are required in P1.

### P5 — Generator validity baseline

A Phase 2 protocol must specify:
- What validity rate is required from the Phase 2 training data generator before training begins
- That the GARCH beta bug is resolved in the Phase 2 generator (not `legacy_compatible` mode)
- That the generator's output is validated against the new vector schema before any model is trained

### P6 — Constrained decoder design intent

A Phase 2 protocol must describe (at a high level, not as implementation code):
- The mathematical constraint the decoder must satisfy
- Whether this is enforced via architecture (e.g., sigmoid outputs, parameter projection), loss penalty, or post-hoc correction
- Why latent interpolation alone does not constitute valid structural composition

---

## What Must NOT Happen Before Phase 2 Protocol Is Approved

To prevent result-chasing, premature implementation, and training-before-protocol:

1. **No Phase 2 VAE model code** — no `vae_constrained.py`, `VAEWithConstraints`, or any class inheriting from or replacing `src/vae.py`
2. **No Phase 2 generator** — no `data_generator_v2.py`, `clean_validated` mode, or generator that produces C-family samples
3. **No Phase 2 training script** — no `train_vae_v2.py`, `train_phase2.py`, or argparse-based training runner
4. **No Phase 2 metric code** — no `metrics.py`, `structural_composition_score`, or `c_validity_rate` implementation
5. **No Phase 2 config file** — no `configs/phase2.yaml` or any config that references Phase 2 components
6. **No Phase 2 tests** — no `test_vae_constrained.py`, `test_generator_v2.py`, or any test for Phase 2 components
7. **No Phase 2 experiment runner** — no `run_phase2.py`, `phase2_reproduce.py`, or similar

---

## Recommended P2 Task Scope (When Authorized)

**TASK_2 label: PHASE_2_PROTOCOL_AND_SCHEMA_DESIGN**

This task is NOT yet authorized. When it is, it should:
1. Write the vector schema v2 document (not code — a specification)
2. Write the family split contract (A/B/C definitions and C exclusion rule)
3. Write the metric contract (success thresholds, validity definitions, zero-shot definition)
4. Write the config discipline specification (which values must be explicit, which flags must change)
5. Write the generator validity baseline specification
6. Write the constrained decoder design intent document
7. Write `PHASE_2_PROTOCOL.md` as a single document summarizing all of the above

**All of 1-7 are documentation only — no code.**

**TASK_3 label: PHASE_2_IMPLEMENTATION** (not yet scoped)

After TASK_2 is complete and approved, a separate task should implement Phase 2 code guided by the protocol document.

---

## P1 Open Items (Still Unresolved)

| Item | Status | Blocking |
|---|---|---|
| torch installation | NOT DONE | Blocks B1 |
| pip freeze / lock file | NOT DONE | Blocks B3 |
| `code_git_commit` auto-capture | NOT IMPLEMENTED | Blocks B2 (future Phase 2 task) |
| Runtime verification of VAE | NOT DONE (torch absent) | Informational — does not block P1 docs |
| Full reproduction run in clean env | NOT DONE | Informational — does not block P1 docs |
| PR review of Phase 2 branch | NOT DONE | Standard process |
