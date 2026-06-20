# Repository Finalization Audit

**Phase:** 1P  
**Date:** 2026-06-20  
**Reproduction anchor:** `results/reproduction_20260620_085314/`

---

## 1. Current Tree Summary

```
.                                  ← Root (POLLUTED — see §9)
├── README.md                      ← STALE — Ukrainian, claims invalidated
├── ROADMAP.md                     ← Partially useful, needs review
├── requirements.txt               ← ACTIVE — well-specified
├── configs/
│   └── demo.yaml                  ← ACTIVE CORE
├── src/
│   ├── config.py                  ← ACTIVE CORE
│   ├── data_generator.py          ← ACTIVE CORE
│   ├── geometry.py                ← ACTIVE CORE
│   ├── io_utils.py                ← ACTIVE CORE
│   ├── reproduce.py               ← ACTIVE CORE (7-stage runner)
│   ├── vae.py                     ← ACTIVE CORE
│   ├── validation.py              ← ACTIVE CORE
│   ├── vector_schema.py           ← ACTIVE CORE
│   ├── train_vae_model.py         ← ACTIVE (legacy entry point, standalone)
│   ├── generate_latent_vectors.py ← LEGACY (standalone, not used by runner)
│   ├── latent_space_interpolation_analysis.py ← LEGACY
│   ├── latent_space_visualization.py ← LEGACY
│   ├── analyze_fisher_metrics.py  ← ARCHIVE CANDIDATE (uses invalidated metrics)
│   ├── analyze_interpolation_norm_vs_validity.py ← ARCHIVE CANDIDATE
│   ├── analyze_interpolation_results.py ← ARCHIVE CANDIDATE
│   ├── analyze_roots_boundary_validity.py ← ARCHIVE CANDIDATE
│   ├── baseline_random_search.py  ← ARCHIVE CANDIDATE
│   ├── compute_fisher_metric.py   ← ARCHIVE CANDIDATE (P0 bug, invalidated)
│   ├── regularized_fisher_metric.py ← ARCHIVE CANDIDATE (invalidated)
│   ├── run_analysis.py            ← ARCHIVE CANDIDATE (wrapper only)
│   ├── run_full_pipeline.py       ← ARCHIVE CANDIDATE
│   ├── slerp_latent_trajectory_analysis.py ← ARCHIVE CANDIDATE
│   ├── generate_enriched_training_data.py ← REVIEW REQUIRED
│   ├── visualize_arma_manifold.py ← ZERO BYTES — DELETE CANDIDATE
│   ├── Симуляція.py               ← LEGACY (non-ASCII filename, not in pipeline)
│   ├── Симуляція_2.py             ← LEGACY (non-ASCII filename, not in pipeline)
│   └── experiments/
│       ├── analyze_beta_diversity.py ← ARCHIVE CANDIDATE
│       ├── backtest_vae_models.py ← LEGACY (not called by Stage 6)
│       ├── inspect_orders_generated.py ← ARCHIVE CANDIDATE
│       ├── robustness_architecture_analysis.py ← LEGACY
│       ├── robustness_architecture_stability_analysis.py ← ZERO BYTES
│       ├── robustness_seed_analysis.py ← ZERO BYTES
│       └── robustness_stability_analysis.py ← ZERO BYTES
├── tests/
│   ├── test_config.py             ← ACTIVE TEST
│   ├── test_data_generator.py     ← ACTIVE TEST
│   ├── test_fail_fast_integrity.py← ACTIVE TEST
│   ├── test_geometry.py           ← ACTIVE TEST
│   ├── test_reproduce_skeleton.py ← ACTIVE TEST
│   ├── test_reproduce_stage1_stage2.py ← ACTIVE TEST
│   ├── test_reproduce_stage3.py   ← ACTIVE TEST
│   ├── test_reproduce_stage4.py   ← ACTIVE TEST
│   ├── test_reproduce_stage5_geometry.py ← ACTIVE TEST
│   ├── test_reproduce_stage6_empirical.py ← ACTIVE TEST
│   ├── test_reproduce_stage7_verdict.py ← ACTIVE TEST
│   ├── test_vae.py                ← ACTIVE TEST
│   ├── test_validation.py         ← ACTIVE TEST
│   └── test_vector_schema.py      ← ACTIVE TEST
├── docs/
│   ├── CLAIM_VERDICT_MATRIX.md    ← ACTIVE DOC
│   ├── FINDINGS_SIMPLE.md         ← ACTIVE DOC (new, Phase 1P)
│   ├── GEOMETRY_VALIDITY_STATUS.md← ACTIVE DOC
│   ├── NUMERICAL_PRECISION.md     ← ACTIVE DOC
│   ├── REPRODUCTION_PLAN.md       ← ACTIVE DOC
│   ├── RESEARCH_INTEGRITY.md      ← ACTIVE DOC
│   ├── WEIGHTS_GOVERNANCE.md      ← ACTIVE DOC
│   ├── INTEGRATION_GUIDE.md       ← REVIEW REQUIRED
│   ├── Phase_I_Summary_Report.md  ← ACTIVE DOC (internal, audit history)
│   ├── latent_manifold_sections.md← REVIEW REQUIRED
│   ├── updated_methodology.md     ← REVIEW REQUIRED
│   ├── temp_edit.md               ← DELETE CANDIDATE (scratch)
│   ├── VAE_Architecture_Technical_Specification.md ← ZERO BYTES
│   ├── # Візуалізація латентного простору VAE.md ← NAMING ISSUE (# in filename)
│   ├── Стаття_Результат_Дослідження.md ← ARCHIVE (legacy article source)
│   └── Стаття_Результат_Дослідження_updated.md ← ARCHIVE (duplicate)
├── models/
│   ├── vae_model_weights.pth      ← ACTIVE MODEL CHECKPOINT (primary)
│   ├── vae_beta01.pth             ← ARCHIVE (legacy robustness sweep)
│   ├── vae_beta05.pth             ← ARCHIVE
│   ├── vae_beta10.pth             ← ARCHIVE
│   ├── vae_beta20.pth             ← ARCHIVE
│   ├── vae_beta50.pth             ← ARCHIVE
│   ├── vae_deeper.pth             ← ARCHIVE (architecture experiment)
│   ├── vae_latent16.pth           ← ARCHIVE (architecture experiment)
│   └── vae_wider.pth              ← ARCHIVE (architecture experiment)
├── data/
│   ├── generated/                 ← ARCHIVE CANDIDATE (legacy generated data)
│   ├── raw/                       ← EMPTY
│   └── processed/                 ← EMPTY
├── results/
│   ├── generation_1/              ← ARCHIVE LEGACY RESULT (invalidated Fisher)
│   ├── generation_2/              ← ARCHIVE LEGACY RESULT
│   └── reproduction_20260620_*/   ← ACTIVE REPRO OUTPUT (keep latest)
├── scripts/
│   └── experiments/               ← ARCHIVE CANDIDATE
└── [ROOT POLLUTION — see §9]
```

---

## 2. Active Pipeline Files

| File | Role |
|---|---|
| `src/config.py` | Config loading, Pydantic schema, integrity guard |
| `src/data_generator.py` | Canonical synthetic dataset generator |
| `src/geometry.py` | Corrected pullback Fisher metric implementation |
| `src/io_utils.py` | `require_file` fail-fast helper |
| `src/reproduce.py` | 7-stage reproduction runner CLI |
| `src/vae.py` | VAE model architecture and checkpoint loader |
| `src/validation.py` | Structural validation and layout classification |
| `src/vector_schema.py` | Parameter layout schema, SSOT for field names |
| `configs/demo.yaml` | Public demo config (demo.yaml) |
| `models/vae_model_weights.pth` | Primary VAE checkpoint |

---

## 3. Active Tests

All 14 test files in `tests/` are active. Latest run: **210 passed in 18.76s**.

---

## 4. Active Docs

| File | Purpose |
|---|---|
| `docs/CLAIM_VERDICT_MATRIX.md` | Claim-by-claim audit matrix |
| `docs/FINDINGS_SIMPLE.md` | Plain-language findings (new) |
| `docs/GEOMETRY_VALIDITY_STATUS.md` | Fisher metric invalidity declaration |
| `docs/NUMERICAL_PRECISION.md` | Precision policy |
| `docs/REPRODUCTION_PLAN.md` | Stage-by-stage reproduction spec |
| `docs/RESEARCH_INTEGRITY.md` | No-fake-data contract |
| `docs/WEIGHTS_GOVERNANCE.md` | Model checkpoint governance |
| `docs/Phase_I_Summary_Report.md` | Internal audit history |

---

## 5. Legacy Code Candidates

| File | Issue |
|---|---|
| `src/compute_fisher_metric.py` | P0 Jacobian bug; invalidated; not in pipeline |
| `src/regularized_fisher_metric.py` | Depends on invalidated `compute_fisher_metric`; not in pipeline; hardcoded Windows absolute path in comment line 1 |
| `src/analyze_fisher_metrics.py` | Loads invalidated `.npy` files; hardcoded `results/generation_1/` paths |
| `src/analyze_interpolation_norm_vs_validity.py` | Hardcoded `results/generation_1/` paths; not in pipeline |
| `src/analyze_interpolation_results.py` | Hardcoded `results/generation_1/` paths |
| `src/analyze_roots_boundary_validity.py` | Hardcoded `results/generation_1/` paths |
| `src/baseline_random_search.py` | Imports from `src.Симуляція_2`; not in pipeline |
| `src/run_full_pipeline.py` | Legacy pipeline entry point, superseded by `reproduce.py` |
| `src/run_analysis.py` | 4-line stub calling legacy scripts |
| `src/slerp_latent_trajectory_analysis.py` | `sys.path.append` hack; not in pipeline |
| `src/latent_space_interpolation_analysis.py` | Used as source for `validation.py` logic but itself not in runner |
| `src/generate_latent_vectors.py` | Module-level `from src.io_utils import require_file` at line 57 (E402); standalone only |
| `src/Симуляція.py` | Non-ASCII filename; not in canonical pipeline; generates `generated_valid_thetas.npy` to root |
| `src/Симуляція_2.py` | Non-ASCII filename; writes `generated_valid_thetas.npy`; not in pipeline |
| `src/experiments/analyze_beta_diversity.py` | `sys.path.insert` hack; non-ASCII import |
| `src/experiments/backtest_vae_models.py` | `allow_pickle=True`; unused `yfinance` import; ambiguous `l` variable |
| `src/experiments/inspect_orders_generated.py` | `allow_pickle=True`; hardcoded filename; 2-line file |

---

## 6. Root Pollution

Files that should not be in the repository root:

| File | Issue |
|---|---|
| `Z_train.npy` | Legacy generated data, should be in `data/` or `archive/` |
| `Z_train_enriched.npy` | Same |
| `train_labels.npy`, `train_labels.csv` | Same |
| `train_labels_enriched.npy` | Same |
| `vae_model_weights.pth` | Duplicate of `models/vae_model_weights.pth` |
| `generated_valid_thetas.npy` | Forbidden file from legacy `Симуляція.py`; should NOT exist at root |
| `orders_generated.npy` | Legacy artifact; source unclear |
| `z_generated.npy`, `z_generated.csv` | Legacy generated data |
| `z_train_ar.npy`, `z_train_ar.csv` | Legacy generated data |
| `z_train_garch.npy`, `z_train_garch.csv` | Legacy generated data |
| `z_interp_1.npy` … `z_interp_5.npy` | Legacy interpolation artifacts |
| `interpolation_detailed_1.csv` … `5.csv` | Legacy results |
| `latent_space_tsne.png` | Legacy plot |
| `latent_space_umap.png` | Legacy plot |
| `coefficients_sums_by_lambda.png` | Legacy plot |
| `norms_and_correlations.png` | Legacy plot |
| `validity_by_lambda.png` | Legacy plot |

Total root pollution: **21 files** that do not belong in root.

---

## 7. Naming Issues

| Issue | Files |
|---|---|
| Non-ASCII (Cyrillic) filenames | `src/Симуляція.py`, `src/Симуляція_2.py`, `docs/Стаття_Результат_Дослідження.md`, `docs/Стаття_Результат_Дослідження_updated.md`, `docs/# Візуалізація латентного простору VAE.md` |
| Leading `#` in filename | `docs/# Візуалізація латентного простору VAE.md` — breaks on most shells |
| Zero-byte files | `src/visualize_arma_manifold.py`, `src/experiments/robustness_architecture_stability_analysis.py`, `src/experiments/robustness_seed_analysis.py`, `src/experiments/robustness_stability_analysis.py`, `docs/VAE_Architecture_Technical_Specification.md` |
| Scratch file left in docs | `docs/temp_edit.md` |
| Duplicate article files | `docs/Стаття_Результат_Дослідження.md` and `docs/Стаття_Результат_Дослідження_updated.md` are byte-for-byte identical (242,689 bytes each) |

---

## 8. Windows Absolute Path Issues

| File | Line | Issue |
|---|---|---|
| `src/regularized_fisher_metric.py` | 1 | `# filepath: c:\Users\user\OneDrive\...` — hardcoded Windows path in comment |

No hardcoded `C:\` in import statements or runtime code was found.  
No absolute paths are used at runtime in the canonical pipeline.

---

## 9. GitHub Hygiene Files

| Item | Status |
|---|---|
| `README.md` | PRESENT but stale; contains invalidated Ukrainian claims |
| `LICENSE` | **MISSING** — P0 for any public GitHub release |
| `.gitignore` | **MISSING** — P0; `results/`, `__pycache__/`, `*.pth` will pollute commits |
| `pyproject.toml` or `setup.py` | MISSING — P1; package is not installable |
| `CHANGELOG.md` | MISSING — P2 |
| `CONTRIBUTING.md` | MISSING — P2 |
| `requirements.txt` | PRESENT and well-specified |

---

## 10. Deep Code Quality Audit

### 10.1 Dead Code

| File | Evidence |
|---|---|
| `src/visualize_arma_manifold.py` | Zero bytes — completely empty |
| `src/experiments/robustness_architecture_stability_analysis.py` | Zero bytes |
| `src/experiments/robustness_seed_analysis.py` | Zero bytes |
| `src/experiments/robustness_stability_analysis.py` | Zero bytes |
| `src/run_analysis.py` | 4 lines; calls legacy scripts that are themselves archive candidates |
| `src/generate_enriched_training_data.py` | Not imported anywhere in the active pipeline |

### 10.2 Ruff Lint Findings (66 total)

Run: `python -m ruff check . --statistics`

| Count | Code | Description | Severity |
|---|---|---|---|
| 35 | F401 | Unused imports | P2 (legacy files), P1 (active files) |
| 11 | E402 | Module-level import not at top | P1 (`src/generate_latent_vectors.py:57`) |
| 6 | E741 | Ambiguous variable name (`l`, `I`) | P2 |
| 5 | F541 | f-string without placeholders | P2 |
| 4 | E701 | Multiple statements on one line | P2 |
| 3 | F841 | Unused local variables | P2 |
| 2 | E722 | Bare `except:` | **P1** (masks all exceptions including `KeyboardInterrupt`) |

**In active pipeline files only:**

| File | Issue | Severity |
|---|---|---|
| `src/data_generator.py:42` | `from typing import Literal` unused | P2 |
| `src/data_generator.py:251` | `AppConfig` imported but unused (local import guard) | P2 (intentional; noqa comment present) |

**Bare `except:` in legacy code:**
- `src/analyze_fisher_metrics.py:68` — silently swallows all errors in Riemannian length loop
- `src/analyze_fisher_metrics.py:96` — silently swallows SVD errors

### 10.3 `allow_pickle` Usage

| File | Line | Context | Severity |
|---|---|---|---|
| `src/experiments/backtest_vae_models.py:48` | `np.load(file_path, allow_pickle=True)` | Loads unknown `.npy` data from file path arg | P1 (legacy, archive candidate) |
| `src/experiments/inspect_orders_generated.py:2` | `np.load("orders_generated.npy", allow_pickle=True)` | Hardcoded filename, trust boundary unclear | P1 (legacy) |
| `src/reproduce.py:312, 584` | `np.load(train_labels_path, allow_pickle=True)` | Loads `train_labels.npy` from repro dir | **P0** — `allow_pickle=True` with a path constructed from config is a trust boundary risk. Labels are integer arrays; `pickle=False` is sufficient and safer. |

### 10.4 `sys.path` Manipulation

| File | Line | Issue | Severity |
|---|---|---|---|
| `src/experiments/analyze_beta_diversity.py:9` | `sys.path.insert(0, ...)` | Legacy hack to import `Симуляція_2` | P1 (archive candidate) |
| `src/slerp_latent_trajectory_analysis.py:6` | `sys.path.append(...)` | Legacy sys.path manipulation | P1 (archive candidate) |

### 10.5 Hardcoded Paths to Legacy Artifacts

All `results/generation_1/` hardcoded paths are in legacy/archive-candidate files:
- `src/analyze_fisher_metrics.py` (×3 paths)
- `src/analyze_interpolation_norm_vs_validity.py` (×6 paths)
- `src/analyze_interpolation_results.py` (×1 path constant)
- `src/analyze_roots_boundary_validity.py` (×3 paths)
- `src/compute_fisher_metric.py` (×2 path constants)
- `src/regularized_fisher_metric.py` (×3 path constants)

None of these appear in the canonical active pipeline (reproduced via `src/reproduce.py`).

### 10.6 Non-ASCII Import Risk

`src/experiments/analyze_beta_diversity.py:10`:
```python
from Симуляція_2 import decode_discrete_structure
```
This import uses a Cyrillic module name. This fails on most CI/CD systems and
non-Ukrainian locale environments. It also bypasses `src.` package prefix.

`src/baseline_random_search.py:2`:
```python
from src.Симуляція_2 import decode_discrete_structure
```
Same risk, with `src.` prefix at least consistent.

### 10.7 Randomness and Seed Discipline

- `src/data_generator.py` uses `np.random` without a configurable seed at
  module level. Training data is intentionally non-deterministic per run.
  This is documented behaviour but means Stage 1 outputs differ between runs.
  *Acceptable* (P3) — document explicitly.
- No `torch.manual_seed` call found in `src/train_vae_model.py`. Acceptable for
  exploratory research; document for reproducibility. (P3)

### 10.8 Async / Concurrency

**No async functions, `asyncio`, `threading`, or `multiprocessing` found
anywhere in `src/` or `tests/`.**  
This is a synchronous, CPU-bound codebase. No concurrency risks.

### 10.9 Network / API Calls

No `requests`, `binance`, `urllib`, `http.client`, or `socket` calls found in
active pipeline files. The legacy `requirements.txt` excludes `python-binance`
intentionally (noted in comment). `yfinance` is imported in
`src/experiments/backtest_vae_models.py:3` but is unused (F401 lint warning).

**No network calls exist in the canonical pipeline.**

### 10.10 Subprocess Usage

No `subprocess` calls found anywhere. Safe.

### 10.11 Logging Quality

- Active pipeline (`src/reproduce.py`) uses `logging.basicConfig` with
  timestamped, levelled output. Good quality.
- Legacy scripts use `print()` statements only. Acceptable for archive-bound code.
- No sensitive data is logged anywhere.

### 10.12 Type Hints

- Active core files (`config.py`, `data_generator.py`, `geometry.py`,
  `validation.py`, `vector_schema.py`, `vae.py`, `reproduce.py`) have
  reasonably complete type hints using Python 3.10+ syntax.
- Some return types use `None` implicitly; mypy would flag these.
- Legacy files have sparse or no type hints.

### 10.13 mypy / pyright Readiness

`mypy` is not configured (no `mypy.ini`, `pyproject.toml`, or `setup.cfg`).
Running `python -m mypy src` would likely report:
- Missing stubs for `numpy`, `torch`, `pydantic` (though pydantic has official stubs).
- Missing `__init__.py` in `src/experiments/`.
- Some implicit `Any` in legacy code.

**Tool not configured: TOOL_NOT_CONFIGURED**

### 10.14 black / ruff Format Readiness

`black` and `ruff` are available but not configured (no `pyproject.toml`).
Ruff found 66 issues. Black would reformat most active files.

**Tool installed but not configured: TOOL_NOT_CONFIGURED**

### 10.15 Other Tools

| Tool | Status |
|---|---|
| `pytest` | CONFIGURED, 210/210 pass |
| `ruff` | INSTALLED, not configured in `pyproject.toml` |
| `black` | INSTALLED (available), not configured |
| `mypy` | TOOL_NOT_CONFIGURED |
| `coverage` | TOOL_NOT_CONFIGURED |
| `vulture` | TOOL_NOT_CONFIGURED |
| `bandit` | TOOL_NOT_CONFIGURED |
| `pip-audit` | TOOL_NOT_CONFIGURED |
| `pre-commit` | MISSING — no `.pre-commit-config.yaml` |

---

## 11. Severity Summary

### P0 — Must Fix Before GitHub Publication

| # | Issue | File | Action |
|---|---|---|---|
| P0-1 | **LICENSE missing** | root | Add `LICENSE` (e.g. MIT or Apache-2.0) |
| P0-2 | **.gitignore missing** | root | Add `.gitignore` covering `__pycache__/`, `*.pyc`, `results/`, `*.pth` in root, `*.npy` in root |
| P0-3 | **README.md contains invalidated claims** | `README.md` | Replace with `docs/README_BLUEPRINT.md` content |
| P0-4 | **`generated_valid_thetas.npy` at root** | root | Verify it is not auto-committed; it is a legacy forbidden artifact |
| P0-5 | **`allow_pickle=True` in `src/reproduce.py:312, 584`** | `src/reproduce.py` | Switch to `allow_pickle=False`; labels are integer arrays |

### P1 — Should Fix Before GitHub Publication

| # | Issue | File | Action |
|---|---|---|---|
| P1-1 | `sys.path` manipulation | `src/experiments/analyze_beta_diversity.py` | Archive the file |
| P1-2 | Non-ASCII module import | `src/experiments/analyze_beta_diversity.py`, `src/baseline_random_search.py` | Archive files |
| P1-3 | `E402` module import not at top | `src/generate_latent_vectors.py:57` | Move import to top (or archive the file) |
| P1-4 | Bare `except:` | `src/analyze_fisher_metrics.py:68, 96` | Archive file; fix if retained |
| P1-5 | `allow_pickle=True` | `src/experiments/backtest_vae_models.py:48` | Archive file |
| P1-6 | `pyproject.toml` missing | root | Add minimal `pyproject.toml` for packaging metadata |
| P1-7 | Zero-byte files committed | 5 files | Delete or remove from git tracking |
| P1-8 | `docs/temp_edit.md` scratch file in docs | `docs/temp_edit.md` | Delete |
| P1-9 | Root pollution (21 artifact files) | root | Move to `archive/` or `data/` per plan |
| P1-10 | Duplicate article files (identical bytes) | `docs/Стаття_*.md` | Archive one; retain one labeled as legacy |

### P2 — Acceptable but Document

| # | Issue | Action |
|---|---|---|
| P2-1 | 35 unused imports (legacy files) | Archive files or add `# noqa: F401` |
| P2-2 | Ambiguous variable names (`l`) | Fix in active code; ignore in archive |
| P2-3 | No seed discipline documented | Add note to `docs/REPRODUCTION_PLAN.md` |
| P2-4 | `CHANGELOG.md` missing | Add before final publication |
| P2-5 | `CONTRIBUTING.md` missing | Add before final publication |
| P2-6 | `# Візуалізація` — `#` in filename | Rename to `latent_space_visualization_notes.md` |

### P3 — Future Cleanup

| # | Issue | Action |
|---|---|---|
| P3-1 | mypy not configured | Add `mypy` to `pyproject.toml` after cleanup |
| P3-2 | ruff not in `pyproject.toml` | Add `[tool.ruff]` config |
| P3-3 | black not in `pyproject.toml` | Add `[tool.black]` config |
| P3-4 | `pre-commit` hooks not configured | Add `.pre-commit-config.yaml` post-cleanup |
| P3-5 | Ukrainian comments in active code | Translate or annotate in a follow-up PR |

---

## 12. GitHub Readiness Checklist

| Item | Status | Notes |
|---|---|---|
| `README.md` ready | ❌ NO | Contains invalidated claims in Ukrainian |
| `LICENSE` present | ❌ NO | P0 blocker |
| `requirements.txt` present | ✅ YES | Well-specified |
| `pyproject.toml` present | ❌ NO | P1 |
| `.gitignore` correct | ❌ NO | Missing entirely; P0 |
| `tests/` pass | ✅ YES | 210/210 |
| Large files policy clear | ⚠️ PARTIAL | `WEIGHTS_GOVERNANCE.md` exists; `.gitignore` not yet enforced |
| Model checkpoint policy clear | ✅ YES | `docs/WEIGHTS_GOVERNANCE.md` documents policy |
| Archive policy clear | ⚠️ PARTIAL | `ARCHIVE_PLAN.md` drafted in Phase 1P |
| No secrets / API keys | ✅ YES | Confirmed by grep scan |
| No fake data | ✅ YES | Integrity config enforced |
| No invalid claims in pipeline | ✅ YES | Stage 7 verdict enforces this |
| Repro command documented | ⚠️ PARTIAL | In `REPRODUCTION_PLAN.md` but README is stale |
| Ukrainian-only filenames risk | ❌ NO | Several Cyrillic filenames; CI risk |

---

## 13. Compile Check

```
python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); import compileall;
           compileall.compile_dir('src', quiet=1); compileall.compile_dir('tests', quiet=1);
           print('compileall OK')"
→ compileall OK
```

Note: native `python -m compileall src tests` fails with `UnicodeEncodeError`
because the Windows terminal codec (cp1252) cannot encode Cyrillic filenames in
the compile status output. This is a display-only issue; the code itself
compiles successfully when UTF-8 is configured.  
**P1 — document and add `PYTHONUTF8=1` to CI environment.**
