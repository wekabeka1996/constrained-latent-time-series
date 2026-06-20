# Quality Gate Commands

**Phase:** 1P  
**Date:** 2026-06-20

This document lists the recommended quality gate commands for this repository,
their status, and the output from Phase 1P execution.

---

## 1. Required Gates (Must Pass Before GitHub)

### 1.1 Full Test Suite

```bash
python -m pytest tests -q
```

**Status:** ✅ PASS  
**Output:**
```
........................................................................ [ 34%]
........................................................................ [ 68%]
..................................................                       [100%]
210 passed in 19.44s
```

### 1.2 Compile Check

```bash
# Native command fails on Windows with Cyrillic filenames in cp1252 terminal.
# Use UTF-8 reconfigured version instead:
python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
import compileall
compileall.compile_dir('src', quiet=1)
compileall.compile_dir('tests', quiet=1)
print('compileall OK')
"
```

**Status:** ✅ PASS (`compileall OK`)

> [!NOTE]
> On Windows, set `PYTHONUTF8=1` in your CI environment to avoid
> `UnicodeEncodeError` when `compileall` attempts to print Cyrillic filenames
> to the console.

### 1.3 End-to-End Reproduction Runner

```bash
python -m src.reproduce --config configs/demo.yaml --stages 1,2,3,4,5,6,7
```

**Status:** ✅ PASS  
**Expected output:**
```
[INFO] Initialized reproduction directory: results\reproduction_<timestamp>
[INFO] Executing Stage 0: Environment & Config Validation
[INFO] Executing Stage 1: Synthetic Methodology Dataset Generation
[INFO] Executing Stage 2: VAE Checkpoint Verification
[INFO] Executing Stage 3: Latent Representation Extraction
[INFO] Executing Stage 4: Structural Validation
[INFO] Executing Stage 5: Corrected Geometry Baseline
[INFO] Executing Stage 6: Empirical Stress Test
[INFO] Executing Stage 7: Final Claim Verdict Report
[INFO] Reproduction run completed successfully.
```

---

## 2. Lint and Format Gates (Configured but Not Enforced)

### 2.1 ruff — Lint Check

```bash
python -m ruff check .
```

**Status:** ⚠️ NOT ENFORCED (66 issues found)  
**Summary:**
```
35  F401  unused-import
11  E402  module-import-not-at-top-of-file
 6  E741  ambiguous-variable-name
 5  F541  f-string-missing-placeholders
 4  E701  multiple-statements-on-one-line-colon
 3  F841  unused-variable
 2  E722  bare-except
Found 66 errors.
[*] 40 fixable with the --fix option.
```

**Note:** Most issues are in legacy/archive-candidate files. Active pipeline
files have ≤3 minor issues. Run `python -m ruff check src/reproduce.py
src/config.py src/data_generator.py src/geometry.py src/validation.py
src/vector_schema.py src/vae.py src/io_utils.py` for active-only check.

**Auto-fix command (safe for active files):**
```bash
python -m ruff check --fix src/config.py src/data_generator.py
```

**Recommended action:** Add `pyproject.toml` with `[tool.ruff]` config to
exclude legacy/archive files from the lint scope.

### 2.2 black — Format Check

```bash
python -m black --check .
```

**Status:** `TOOL_NOT_CONFIGURED` (not in `pyproject.toml`)

**Recommended action:** Add `[tool.black]` to `pyproject.toml` after
archive cleanup, then run `python -m black src/ tests/`.

---

## 3. Type Checking Gates (Not Configured)

### 3.1 mypy

```bash
python -m mypy src
```

**Status:** `TOOL_NOT_CONFIGURED` (no `mypy.ini` or `pyproject.toml` section)

**Expected issues if run:**
- Missing stubs for `numpy`, `torch` (third-party stub packages needed)
- Some `Any` types in legacy code
- Missing `__init__.py` in `src/experiments/`

**Recommended action:** After archive cleanup, add:
```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_return_any = false
```

---

## 4. Coverage Gate (Not Configured)

```bash
python -m pytest --cov=src tests -q
```

**Status:** `TOOL_NOT_CONFIGURED`  
**Note:** `pytest-cov` must be installed: `pip install pytest-cov`

---

## 5. Security Gates (Not Configured)

### 5.1 bandit — Security Audit

```bash
python -m bandit -r src
```

**Status:** `TOOL_NOT_CONFIGURED`

**Known findings (from manual audit):**
- `allow_pickle=True` in `src/reproduce.py:312, 584` — P0 trust boundary risk
- `allow_pickle=True` in legacy experiment files — archive candidates

### 5.2 pip-audit — Dependency Vulnerability Scan

```bash
python -m pip_audit
```

**Status:** `TOOL_NOT_CONFIGURED`

**Recommended action:** Add to CI pipeline.

---

## 6. Secret Scanning

Manual scan performed in Phase 1P using grep patterns:
```
API_KEY, SECRET, TOKEN, BINANCE, OPENAI, PRIVATE, PASSWORD
```

**Status:** ✅ No secrets found in any tracked file.

---

## 7. Pre-commit Hooks (Not Configured)

No `.pre-commit-config.yaml` exists. Recommended configuration after cleanup:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files
        args: ['--maxkb=5000']
```

---

## 8. CI/CD Recommended Pipeline

When publishing to GitHub, add a `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
env:
  PYTHONUTF8: "1"
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python -m pytest tests -q
      - run: python -m ruff check src/ tests/
```

**Note:** The Cyrillic filename `src/Симуляція.py` etc. will already be in
`archive/` by the time this CI runs (Phase 1Q cleanup). If they are still
present, the `ruff check` scope should be restricted to `src/config.py` etc.

---

## 9. P0 Fix Commands (Before GitHub)

```bash
# Fix allow_pickle in reproduce.py (lines 312 and 584)
# Change: np.load(..., allow_pickle=True) for integer label arrays
# To:     np.load(...)
# → Edit src/reproduce.py manually or use ruff/sed

# Create .gitignore
# → See docs/ARCHIVE_PLAN.md for recommended .gitignore content

# Add LICENSE
# → Choose MIT or Apache-2.0; create LICENSE file at root

# Update README.md
# → Replace with content from docs/README_BLUEPRINT.md after user approval
```
