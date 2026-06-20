# PHASE_2_P1_REVIEW_FIX_NOTES.md
# Phase 2 P1: Review Fix Notes

**Date:** 2026-06-20
**Branch:** phase2/p1-repo-inventory
**Commit (at start of fix):** 67f06e142bc0cab191ef22067f84e74e3c8755fe
**Purpose:** Document each reviewer finding, the fix applied, and the evidence gathered.

---

## Reviewer Problem 1 — Missing Git Evidence

**Problem stated:** Previous final report did not provide commit hash, push confirmation,
or remote branch verification. Git evidence was absent from all three P1 reports.

**Fix applied:**

The following commands were run and their outputs are now embedded in the reports:

```powershell
Command: git branch --show-current
Output:  phase2/p1-repo-inventory

Command: git log --oneline -1
Output:  67f06e1 PHASE2_P1 repo inventory and failure map

Command: git log --oneline -5
Output:
  67f06e1 PHASE2_P1 repo inventory and failure map
  04cd4ca Polish public README and publication hygiene
  8d0af88 Finalize reproducible VAE latent-geometry research repo

Command: git status --short
Output:  (empty — clean working tree)

Command: git ls-files reports/
Output:
  reports/PHASE_2_P1_FAILURE_MAP.md
  reports/PHASE_2_P1_NEXT_STEPS.md
  reports/PHASE_2_P1_REPO_INVENTORY.md

Command: git ls-remote origin phase2/p1-repo-inventory
Output:  67f06e142bc0cab191ef22067f84e74e3c8755fe  refs/heads/phase2/p1-repo-inventory
```

Full commit hash: `67f06e142bc0cab191ef22067f84e74e3c8755fe`

This confirms:
- The branch exists locally and remotely
- The remote branch tip matches the local HEAD
- The three report files are tracked by git

**Report updated:** All three reports now display the full commit hash and branch name in their headers.
**PR link:** `https://github.com/wekabeka1996/constrained-latent-time-series/pull/new/phase2/p1-repo-inventory`

---

## Reviewer Problem 2 — C Leakage Framing Was Wrong

**Problem stated:** The v1 failure map incorrectly described ARMA-only examples
(from `generate_enriched_training_data.py`) as "C leakage." Under the Phase 2 benchmark intent
(A=AR/ARMA, B=GARCH, C=ARMA-GARCH), ARMA belongs to the A-family, not C.

**Fix applied:**

A "Corrected Framing" note has been added to both the Repo Inventory (§4 generator section)
and the Failure Map (§3 C Leakage Risk):

- The correct Phase 2 family definitions are stated explicitly:
  - A = AR / ARMA (mean dynamics only)
  - B = GARCH (volatility dynamics only)
  - C = ARMA-GARCH (combined mean + volatility hybrid)
- The correct definition of C-leakage is stated: C-leakage occurs when ARMA-GARCH models appear in training data
- The current risk is correctly restated as **absence of a formal split contract and schema-level family guard**, not confirmed C-leakage
- The previous framing ("ARMA in training = C leakage") has been removed from both reports

---

## Reviewer Problem 3 — Hardcoded/Default Audit Was Incomplete

**Problem stated:** The previous reports did not show actual command output for the defaults audit.
The reviewer required: command shown, output shown, and machine-specific paths removed.

**Fix applied:**

The following PowerShell command was run:

```powershell
Select-String -Path "src\*.py","tests\*.py","configs\*.yaml","pyproject.toml","README.md" `
  -Pattern "default=|seed|latent_dim|d_latent|hidden|threshold|epsilon|n_arma|n_garch|output_dir|
            checkpoint_path|allow_cached|np\.save|torch\.save|argparse|dataclass|BaseModel" `
  -CaseSensitive:$false | ForEach-Object { "$($_.Filename):$($_.LineNumber): $($_.Line.Trim())" }

Note: ripgrep (rg) was not available. Select-String was used instead.
```

The full output from `src/` only was extracted, including:
- `src/config.py:87,124,147,148,149,165` — all Pydantic defaults
- `src/data_generator.py:66,67,68` — hardcoded n_arma, n_garch, seed
- `src/generate_enriched_training_data.py:47` — bare literal `n_arma = 100`
- `src/reproduce.py:39` — argparse default config path
- `src/train_vae_model.py:11,64,65,66,67,68,69` — module constant D_LATENT and all argparse defaults
- `src/vae.py:42,43,44,85,86` — module constants D_LATENT, D_HIDDEN1, D_HIDDEN2
- `src/validation.py:231,270,309,358,359` — all tol, threshold, epsilon function defaults

These are now documented in:
- `PHASE_2_P1_FAILURE_MAP.md` §4 (full table with file:line references and Phase 2 action)
- `PHASE_2_P1_REPO_INVENTORY.md` §5 (config defaults table for `configs/demo.yaml` + Pydantic gaps)

Machine-specific absolute paths have been removed from all reports. File references use
repository-relative paths throughout.

---

## Reviewer Problem 4 — Missing Test Evidence

**Problem stated:** Previous reports did not show which tests pass, which skip, and which fail.
The reviewer required: command shown, output shown, and failure reasons documented.

**Fix applied:**

The following commands were run:

```powershell
Command: python -m pytest tests/test_config.py tests/test_data_generator.py
         tests/test_validation.py tests/test_vector_schema.py -q --no-header
Output:  84 passed, 4 skipped in 0.34s

Command: python -m pytest tests/ --ignore=[9 torch-dependent files] -q --no-header --tb=short
Output:  91 passed, 4 skipped in 0.28s
         (91 = all torch-free tests across all runnable test files)

Command: python -m pytest tests/ -q --no-header --tb=no
Output:
  ERROR tests/test_geometry.py
  ERROR tests/test_reproduce_skeleton.py
  ERROR tests/test_reproduce_stage1_stage2.py
  ERROR tests/test_reproduce_stage3.py
  ERROR tests/test_reproduce_stage4.py
  ERROR tests/test_reproduce_stage5_geometry.py
  ERROR tests/test_reproduce_stage6_empirical.py
  ERROR tests/test_reproduce_stage7_verdict.py
  ERROR tests/test_vae.py
  9 errors in 0.53s
  (All errors are: ModuleNotFoundError: No module named 'torch')

Environment verified:
  python --version → Python 3.14.3
  pip show numpy   → Version: 2.4.2
  pip show pydantic → Version: 2.12.3
  pip show pyyaml  → Version: 6.0.3
  pip show torch   → WARNING: Package(s) not found: torch
```

This evidence is now embedded in:
- `PHASE_2_P1_REPO_INVENTORY.md` §2 (Test Evidence header block)
- `PHASE_2_P1_REPO_INVENTORY.md` §6 (full test inventory with two tables: runnable vs. torch-dependent)

The 4 skips are explained: they are deliberate skip conditions triggered by missing legacy root
artifacts (`Z_train.npy`, `train_labels.npy`, `generated_valid_thetas.npy`) which are
gitignored and expected to be absent.

---

## Reviewer Problem 5 — Next Steps Described Implementation, Not Documentation

**Problem stated:** The v1 NEXT_STEPS report described code to implement (`vector_schema_v2.py`,
`data_generator_v2.py`, etc.), violating the P1 documentation-only constraint.

**Fix applied:**

The following changes were made to `PHASE_2_P1_NEXT_STEPS.md`:

1. Added a prominent P1-is-still-active warning at the top
2. Replaced implementation prescriptions with documentation requirements — each item now describes
   what a future Phase 2 protocol document must address, not what code to write
3. Added a "What Must NOT Happen Before Phase 2 Protocol Is Approved" section with explicit prohibitions:
   - No `vae_constrained.py`
   - No `data_generator_v2.py`
   - No `train_vae_v2.py`
   - No `metrics.py`
   - No `configs/phase2.yaml`
   - No new test files for Phase 2 components
   - No Phase 2 experiment runner
4. Renamed the "Recommended P1 Follow-up Tasks" section to "Recommended P2 Task Scope (When Authorized)"
   and marked it explicitly as NOT YET AUTHORIZED
5. Separated blocker items (B1-B3) from protocol preconditions (P1-P6) for clarity

---

## Files Changed in This Fix Session

```
reports/PHASE_2_P1_REPO_INVENTORY.md    — UPDATED (v2)
reports/PHASE_2_P1_FAILURE_MAP.md       — UPDATED (v2)
reports/PHASE_2_P1_NEXT_STEPS.md       — UPDATED (v2)
reports/PHASE_2_P1_REVIEW_FIX_NOTES.md — NEW
```

## Files NOT Changed in This Fix Session

No source files, test files, config files, model files, archive files, docs files,
or any other file outside of `reports/` was modified.

```powershell
# Verification:
Command: git status --short
Output (after report writes, before commit):
  M reports/PHASE_2_P1_REPO_INVENTORY.md
  M reports/PHASE_2_P1_FAILURE_MAP.md
  M reports/PHASE_2_P1_NEXT_STEPS.md
  ? reports/PHASE_2_P1_REVIEW_FIX_NOTES.md
```

## Remaining Known Limitations (Not Fixed Here)

1. **torch not installed** — runtime VAE behavior is MISSING_EVIDENCE in all reports
2. **No pip freeze / lock file** — environment reproducibility is not captured
3. **`code_git_commit: None` in manifest** — auto-capture not implemented (Phase 2 implementation task)
4. **No full reproduction run** — no `python -m src.reproduce ...` was executed in this environment
