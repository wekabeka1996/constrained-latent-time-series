# PHASE_2_P1_REPO_INVENTORY.md
# Phase 2 P1: Repository Inventory

**Report version:** 2 (post-review fix)
**Date:** 2026-06-20
**Branch:** phase2/p1-repo-inventory
**Commit:** 67f06e142bc0cab191ef22067f84e74e3c8755fe (initial P1 commit)
**Remote branch confirmed:** `refs/heads/phase2/p1-repo-inventory` → `67f06e142bc0cab191ef22067f84e74e3c8755fe`

---

## Git Evidence

```
Command: git branch --show-current
Output:  phase2/p1-repo-inventory

Command: git log --oneline -1
Output:  67f06e1 PHASE2_P1 repo inventory and failure map

Command: git status --short
Output:  (empty — clean working tree at time of initial P1 commit)

Command: git ls-remote origin phase2/p1-repo-inventory
Output:  67f06e142bc0cab191ef22067f84e74e3c8755fe   refs/heads/phase2/p1-repo-inventory

PR link: https://github.com/wekabeka1996/constrained-latent-time-series/pull/new/phase2/p1-repo-inventory
```

---

## Test Evidence

```
Command: python -m pytest tests/test_config.py tests/test_data_generator.py
         tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py -q
Result:  91 passed, 4 skipped in 0.28s
         (4 skips = missing legacy root artifacts Z_train.npy, train_labels.npy,
          generated_valid_thetas.npy — expected and documented)

Full suite (tests/):
  - 9 test files FAILED TO COLLECT due to: ModuleNotFoundError: No module named 'torch'
  - Affected files: test_geometry.py, test_reproduce_skeleton.py,
    test_reproduce_stage1_stage2.py, test_reproduce_stage3.py,
    test_reproduce_stage4.py, test_reproduce_stage5_geometry.py,
    test_reproduce_stage6_empirical.py, test_reproduce_stage7_verdict.py,
    test_vae.py

Environment:
  Python:  3.14.3
  numpy:   2.4.2
  pydantic: 2.12.3
  PyYAML:  6.0.3
  torch:   NOT INSTALLED
```

> [!WARNING]
> **KNOWN LIMITATION — MISSING TEST ENVIRONMENT**: `torch` is not installed in the local interpreter.
> Nine test files that depend on PyTorch (VAE loading, encode/decode, checkpoint verification)
> could not be collected or executed. All findings in those areas are based on **static code audit
> only**, not runtime verification. These are marked MISSING_EVIDENCE where applicable.

---

## 1. Repository Overview

### Top-Level Tree Summary

```
.                                        ← Repository root
├── .github/
│   └── workflows/ci.yml
├── archive/                             ← Historical/invalidated code and results
│   ├── invalidated_geometry/            ← P0-bug scripts (Fisher metric computation)
│   ├── legacy_code/                     ← Pre-pipeline standalone scripts
│   ├── legacy_results/generation_2/    ← Archived generation-2 robustness results
│   ├── old_experiments/                 ← Archived experiment scripts
│   ├── original_research/              ← Original article (contains invalidated claims)
│   ├── README_ARCHIVE.md
│   └── latent_space_visualization_notes.md
├── configs/
│   └── demo.yaml                        ← Single active config; all others MISSING
├── docs/                                ← Active research documents
│   ├── evidence/
│   │   ├── README.md
│   │   └── publication_summary.json
│   ├── ARCHIVE_PLAN.md
│   ├── CLAIM_VERDICT_MATRIX.md
│   ├── FILE_CLASSIFICATION_MANIFEST.md
│   ├── FINDINGS_SIMPLE.md
│   ├── GEOMETRY_VALIDITY_STATUS.md
│   ├── INTEGRATION_GUIDE.md
│   ├── NUMERICAL_PRECISION.md
│   ├── Phase_I_Summary_Report.md
│   ├── QUALITY_GATE_COMMANDS.md
│   ├── README_BLUEPRINT.md
│   ├── REPO_FINALIZATION_AUDIT.md
│   ├── REPRODUCTION_PLAN.md
│   ├── RESEARCH_INTEGRITY.md
│   ├── WEIGHTS_GOVERNANCE.md
│   ├── latent_manifold_sections.md
│   └── updated_methodology.md
├── models/
│   ├── vae_beta50.pth                   ← Public demo checkpoint (Git-tracked)
│   └── vae_model_weights.pth            ← Retained legacy checkpoint (Git-tracked)
├── reports/                             ← Phase 2 P1 documentation (this folder)
│   ├── PHASE_2_P1_REPO_INVENTORY.md     ← This file
│   ├── PHASE_2_P1_FAILURE_MAP.md
│   ├── PHASE_2_P1_NEXT_STEPS.md
│   └── PHASE_2_P1_REVIEW_FIX_NOTES.md
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── generate_enriched_training_data.py  ← NOT in canonical pipeline
│   ├── geometry.py
│   ├── io_utils.py
│   ├── reproduce.py                     ← 7-stage reproduction runner (CLI entry)
│   ├── train_vae_model.py               ← Legacy standalone training script
│   ├── vae.py
│   ├── validation.py
│   └── vector_schema.py
├── tests/                               ← 14 test files; 5 runnable without torch
│   └── [14 test files — see test inventory]
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── ROADMAP.md
└── requirements.txt
```

### Directories by role

| Role | Path |
|---|---|
| Main source | `src/` |
| Config | `configs/` |
| Tests | `tests/` |
| Documentation | `docs/` |
| P1 reports | `reports/` |
| Archive (non-pipeline) | `archive/` |
| CLI entry points | `src/reproduce.py`, `src/train_vae_model.py` |
| Notebooks | **None** |
| Generated data (local-only) | `results/reproduction_YYYYMMDD_*/` — ignored by `.gitignore` |
| Results tracked in Git | **None** (`.npy`, `.csv`, `.png`, `results/reproduction_*/` all gitignored) |

---

## 2. Current Project Purpose

The repository is a **research reproducibility and integrity study** — not a production system, trading tool, or demo application. It evaluates whether a VAE can learn a continuous latent manifold over discrete econometric model families (AR/ARMA parameters and GARCH parameters).

**Pipeline classification:** Research prototype with 7-stage auditable runner.

**Phase 1 outcome (Phase 1P as documented):**
- Encoder-side clustering: PARTIALLY_CONFIRMED
- Decoder structural validity: INVALIDATED (0/200 = 0.0% valid reconstructions, all `LEGACY_LAYOUT_COLLISION`)
- Interpolation validity: INVALIDATED
- Legacy Fisher geometry: INVALIDATED (P0 Jacobian bug)
- Empirical stress test: NEEDS_REAL_DATA
- Reconstruction validity claims (~100%): INVALIDATED

---

## 3. Existing Models Inventory

### Active Model: `src/vae.py` — class `VAE`

| Attribute | Value |
|---|---|
| Path | `src/vae.py` |
| Class | `VAE` |
| Architecture | Encoder: Linear(40→64)→ReLU→Linear(64→32)→ReLU; μ,logσ² projections; Decoder: Linear(8→32)→ReLU→Linear(32→64)→ReLU→Linear(64→40)→Tanh |
| Input dim | 40 (hardcoded in module constants `D_INPUT = 40`) |
| Latent dim | 8 (hardcoded in module constant `D_LATENT = 8`) |
| Output dim | 40; values in `[-1, 1]` via Tanh |
| Structure handling | Numeric vector only — no architectural enforcement of structural validity |
| Structural constraint | **None** — decoder does not enforce disjoint indicator activation |
| Phase 2 relevance | High — must be redesigned or extended with constrained decoder head |

**MISSING_EVIDENCE**: VAE forward pass runtime behavior not verified (torch not installed).

### Duplicate Model: `src/train_vae_model.py` — class `VAE` (inline copy)

| Attribute | Value |
|---|---|
| Path | `src/train_vae_model.py` |
| Architecture | Same as `src/vae.py` (inline duplicate) |
| Differences | Different default seed/epoch values; standalone argparse CLI; writes output to hardcoded root path |
| Phase 2 risk | **Duplication risk** — two diverging definitions of `VAE`; standalone script bypasses config discipline |

### Checkpoint files

| File | Role | Git-tracked |
|---|---|---|
| `models/vae_beta50.pth` | Public demo default (referenced in `configs/demo.yaml`) | Yes |
| `models/vae_model_weights.pth` | Retained legacy checkpoint | Yes |

**MISSING_EVIDENCE**: Checkpoint SHA-256 values verified at runtime — not reproducible here because torch is absent.

---

## 4. Existing Generator/Data Inventory

### Active Generator: `src/data_generator.py`

| Attribute | Value |
|---|---|
| Generated families | A = AR-like (labeled "AR"), B = GARCH (labeled "GARCH") |
| C / hybrid structures | **None generated** — strictly disjoint by design |
| C in training | **No** — `allow_supervised_arma_garch: false` enforced |
| AR coefficient sampling | `phi` ∈ Uniform(-0.8, 0.8) per coefficient |
| GARCH omega sampling | ω ∈ Uniform(0.1, 0.5) |
| GARCH alpha sampling | α ∈ Uniform(0.0, 0.3) per term |
| GARCH beta sampling | β ∈ Uniform(min(0, 0.4-Σα), max(0, 0.4-Σα)) — **KNOWN BUG: negative beta when Σα > 0.4** |
| Labels | String array: "AR" or "GARCH" |
| Ground-truth metadata | Encoded in first 10 dims of each vector (struct slice); separate label array |
| Legacy mode | `legacy_compatible` — reproduces original GARCH beta bug intentionally for checkpoint compatibility |
| Clean mode | `clean_validated` — **NOT YET IMPLEMENTED** |
| Hardcoded defaults in class | `n_arma = 100`, `n_garch = 100`, `seed = 42`, `max_lag_order = 5` |

### Non-canonical Script: `src/generate_enriched_training_data.py`

| Attribute | Value |
|---|---|
| Generated families | ARMA (via root-placement method) |
| Layout used | **Non-canonical** — uses raw integer orders at indices 3,4 instead of continuous logits; places AR at `v[10]`, MA at `v[20]` in a non-standard offset |
| Labels | Integer `2` appended to string "AR"/"GARCH" labels — **type collision** |
| Pipeline status | **Not part of canonical pipeline** — standalone script, reads/writes root-level `.npy` files |
| C-family status | Generates ARMA models — whether these constitute "C" depends on Phase 2 protocol definition |

> [!IMPORTANT]
> **Note on C leakage framing**: Under the Phase 2 benchmark intent (A=AR/ARMA, B=GARCH, C=ARMA-GARCH), ARMA-only models belong to the A-family or A-subfamily. True C-leakage would require ARMA-GARCH hybrid (mean + volatility) models appearing in zero-shot training data. The current risk is **absence of a formal split contract and schema-level family guard**, not confirmed C-leakage.

---

## 5. Existing Config Inventory

### Single config: `configs/demo.yaml`

```
Command: git ls-files configs/
Output:  configs/demo.yaml
```

| Section | Key | Value | Explicit/Implicit |
|---|---|---|---|
| project | name | `econometric-vae-manifold` | Explicit |
| project | mode | `demo` | Explicit |
| model | architecture | `baseline_vae` | Explicit |
| model | input_dim | `40` | Explicit |
| model | latent_dim | `8` | Explicit |
| model | hidden_dims | `[64, 32]` | Explicit |
| model | checkpoint_path | `models/vae_beta50.pth` | Explicit |
| model | checkpoint_role | `public_demo_default` | Explicit |
| vector_schema | vector_dim | `40` | Explicit |
| vector_schema | struct_dim | `10` | Explicit |
| vector_schema | param_dim | `20` | Explicit |
| vector_schema | stat_dim | `10` | Explicit |
| vector_schema | max_lag_order | `5` | Explicit |
| generation | n_arma | `100` | Explicit |
| generation | n_garch | `100` | Explicit |
| generation | seed | `42` | Explicit |
| generation | mode | `legacy_compatible` | Explicit |
| generation | allow_supervised_arma_garch | `false` | Explicit |
| generation | synthetic_training_data_is_methodology | `true` | Explicit |
| validation | activation_threshold | `0.5` | Explicit |
| validation | activation_epsilon | `0.05` | Explicit |
| validation | tolerance | `1.0e-8` | Explicit |
| paths | market_data_csv | `null` | Explicit |
| paths | output_dir | `results/demo` | Explicit |
| integrity | allow_random_fallbacks | `false` | Explicit |
| integrity | allow_fake_market_data | `false` | Explicit |
| integrity | allow_cached_artifacts | `true` | Explicit |
| integrity | require_artifact_provenance | `true` | Explicit |
| integrity | fail_fast_on_missing_required_artifacts | `true` | Explicit |
| geometry | (all fields) | **ABSENT from demo.yaml** | **Implicit via Pydantic defaults** |

**Pydantic defaults NOT covered by demo.yaml** (from `src/config.py`):

```
src/config.py:147: max_geometry_points: int = Field(default=32, gt=0)
src/config.py:148: finite_difference_eps: float = Field(default=1e-4, gt=0.0)
src/config.py:149: metric_regularization_eps: float = Field(default=1e-8, gt=0.0)
src/config.py:164: integrity: IntegrityConfig = IntegrityConfig()   ← entire block defaults silently
src/config.py:165: geometry:  GeometryConfig  = GeometryConfig()    ← entire block defaults silently
```

> [!CAUTION]
> **Config discipline gap**: The `geometry` block is entirely absent from `configs/demo.yaml`.
> Its three research-affecting values (`max_geometry_points=32`, `finite_difference_eps=1e-4`,
> `metric_regularization_eps=1e-8`) are silently populated by Pydantic defaults.
> Phase 2 must not permit this pattern — every research-affecting value must be explicit in
> the Phase 2 config and must fail-fast if absent.

**Other silent defaults catalogued (not "frozen" — documented as legacy values requiring explicit Phase 2 override):**

| Location | Default | Affects research |
|---|---|---|
| `src/config.py:87` | `seed: int = 42` | Yes — data generation |
| `src/config.py:124` | `allow_cached_artifacts: bool = True` | Yes — provenance |
| `src/data_generator.py:66-68` | `n_arma=100, n_garch=100, seed=42` | Yes — dataset size and reproducibility |
| `src/validation.py:231,270,309` | `tol: float = 1e-8` | Yes — validity boundaries |
| `src/validation.py:358-359` | `threshold=0.5, epsilon=0.05` | Yes — family classification |
| `src/vae.py:43-44` | `D_LATENT=8, D_HIDDEN1=64, D_HIDDEN2=32` | Yes — model capacity |
| `src/train_vae_model.py:64-69` | `epochs=100, seed=42, beta=1.0`, hardcoded root paths | Yes — training behavior |
| `src/reproduce.py:39` | `default="configs/demo.yaml"` | Yes — config path fallback |

---

## 6. Existing Tests Inventory

### Test files that run without torch (91 passed, 4 skipped)

| File | What it covers | Key gaps |
|---|---|---|
| `tests/test_config.py` | YAML loading, Pydantic validation, dim checks | Does not test missing geometry block |
| `tests/test_data_generator.py` | Shape, labels, determinism, GARCH bug | Does not test validity rate of generated data |
| `tests/test_validation.py` | AR roots, MA roots, GARCH constraints, structure decode | Does not test struct decode with ambiguous thresholds |
| `tests/test_vector_schema.py` | Vector slices, dim assertions | Does not test collision at `v[10]` |
| `tests/test_fail_fast_integrity.py` | Integrity flag enforcement | Does not test silent Pydantic geometry defaults |

### Test files that fail to collect (torch missing)

| File | Reason | What it would cover |
|---|---|---|
| `tests/test_vae.py` | `import torch` | VAE forward/encode/decode |
| `tests/test_geometry.py` | `from src.vae import VAE` → `import torch` | Jacobian, pullback metric |
| `tests/test_reproduce_skeleton.py` | `from src.reproduce import main` → `import torch` | Runner CLI skeleton |
| `tests/test_reproduce_stage1_stage2.py` | `import torch` | Stages 1-2 integration |
| `tests/test_reproduce_stage3.py` | `import torch` | Stage 3 latent extraction |
| `tests/test_reproduce_stage4.py` | `import torch` | Stage 4 structural validation |
| `tests/test_reproduce_stage5_geometry.py` | `import torch` | Stage 5 geometry |
| `tests/test_reproduce_stage6_empirical.py` | `import torch` | Stage 6 empirical |
| `tests/test_reproduce_stage7_verdict.py` | `import torch` | Stage 7 verdict |

### What is NOT covered by any current test

- Split leakage (no test verifies training data excludes a held-out family)
- Generator validity rate (tests assert bug exists, not that outputs are valid)
- Config discipline for missing `geometry` block (silent default not tested)
- Family guard (no test verifies that A/B/C boundary is explicit and enforced)

---

## 7. Existing Reports/Artifacts Inventory

| File | Status | Reproducible |
|---|---|---|
| `docs/CLAIM_VERDICT_MATRIX.md` | Active | Yes (logic-reproducible; evidence from reproduction run) |
| `docs/GEOMETRY_VALIDITY_STATUS.md` | Active | Yes (documents invalidation, not a positive claim) |
| `docs/FINDINGS_SIMPLE.md` | Active | Yes |
| `docs/Phase_I_Summary_Report.md` | Active (audit history) | Partially (tables reference local result paths not in repo) |
| `docs/evidence/publication_summary.json` | Active | Yes |
| `archive/legacy_results/*` | Historical — DO NOT CITE | No (local-only binaries excluded) |
| `results/reproduction_20260620_*/` | Local-only (gitignored) | Reproducible locally via runner command |

### Negative results preserved
- 0/200 = 0.0% reconstruction validity (LEGACY_LAYOUT_COLLISION)
- 125/200 = 62.5% input validity (GARCH beta bug + AR stationarity)
- Legacy Fisher/geodesic metrics: INVALID_DO_NOT_CLAIM

---

## 8. Reproducibility Inventory

| Item | Status | Notes |
|---|---|---|
| Git commit hash | Tracked | `67f06e142bc0cab191ef22067f84e74e3c8755fe` (P1 reports commit) |
| Seed | Explicit in config | `generation.seed: 42` in `configs/demo.yaml` |
| Config snapshot | SHA-256 in manifest | Manifest stores config file hash per run |
| Environment/dependencies | `requirements.txt` | No lock file (e.g. `pip freeze` or `poetry.lock`) — **gap** |
| Run ID | Per-run UUID | Generated by `reproduce.py` |
| Metrics JSON | Per-run | `structural_validation.json`, `corrected_geometry.json` |
| Verdict JSON | Per-run | `final_claim_verdict.json` |
| Generated samples | Per-run | `Z_train.npy`, `reconstructed_theta.npy` in timestamped dirs |
| Model checkpoint hash | SHA-256 | Logged in `checkpoint_verification.json` and manifest |
| Git commit in manifest | `None` hardcoded | `code_git_commit: None` in `build_initial_manifest()` — **gap** |

---

## Known Limitations

1. **MISSING_EVIDENCE — torch not installed**: 9/14 test files could not be collected or executed. Runtime behavior of VAE encoder, decoder, checkpoint loading, and all stage-runner integration tests is based on static code audit only.
2. **MISSING_EVIDENCE — no `pip freeze` / lock file**: The `requirements.txt` specifies minimum versions but not exact installed versions. Full environment reproducibility is not guaranteed.
3. **MISSING_EVIDENCE — runtime reproduction run**: No full `python -m src.reproduce --config configs/demo.yaml --stages 1,2,3,4,5,6,7` was executed in this environment. The publication anchor run (`results/reproduction_20260620_085314/`) is local-only and not in the repository.
4. **Static audit only for archived scripts**: Scripts in `archive/` were inspected by filename and content but not executed.
5. **`code_git_commit: None` in manifest**: The reproduction runner does not auto-capture the git commit hash at runtime; this is hardcoded as `None` in `build_initial_manifest()`.

---

## P1 Acceptance Status

**P1_BLOCKED_BY_MISSING_TEST_ENV**

All three P1 reports have been updated with command evidence, git evidence, and corrected framing. The block is specifically:
- torch not installed → 9 test files cannot be collected or run
- Findings from those 9 files are MISSING_EVIDENCE (static audit only)
- This does not block the documentation phase, but must be declared before any runtime claim is made about VAE behavior

Once the torch environment is available, run:
```
python -m pytest tests/ -q
```
Expected baseline: 200 passed, 4 skipped (per README claim).

---

## Changed Files (this revision)
- `reports/PHASE_2_P1_REPO_INVENTORY.md` — full rewrite with git evidence, test evidence, command output, corrected local path links, honest known limitations, defaults audit table, C-leakage framing fix
- `reports/PHASE_2_P1_FAILURE_MAP.md` — corrected framing, added command evidence, removed machine-specific links, hardened defaults audit
- `reports/PHASE_2_P1_NEXT_STEPS.md` — corrected Prompt 2 scope to documentation-only, removed premature code prescriptions
- `reports/PHASE_2_P1_REVIEW_FIX_NOTES.md` — new file documenting each reviewer finding and fix applied

## Confirmation: No Source Code Changed
No files in `src/`, `tests/`, `configs/`, `models/`, `archive/`, or `docs/` were modified.
Only files in `reports/` were created or updated.
