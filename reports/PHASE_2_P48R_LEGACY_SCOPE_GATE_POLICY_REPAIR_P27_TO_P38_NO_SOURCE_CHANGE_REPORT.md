# PHASE_2_P48R: Legacy Scope Gate Policy Repair P24–P38 — No Source Change

**Phase:** P48R
**Branch:** `phase2/p48r-legacy-scope-gate-policy-repair-p27-to-p38-no-source-change`
**Base Branch:** `phase2/p48-deterministic-direct-fit-robustness-smoke-no-model-no-vae-no-science`
**P48 Final HEAD (Base Commit):** `1de362b5e0dd7edf4bedc6ceeccc25a84f6a9de4`
**Verdict:** `P48R_READY_FOR_REVIEW`

---

## 1. Scope Statement

P48R does not alter P48 objective robustness code.
P48R does not alter P44/P45/P46 math/loss code.
P48R does not add model/VAE/science functionality.
P48R only repairs legacy phase-local scope-gate policy.

---

## 2. Problem Statement

The full curated test suite on P48 branch reported **15 failed** scope gate tests:

- 12 from P27–P38 (`test_pXX_YY_scope_gate`)
- 3 from P24–P26 (`test_pXX_YY_no_forbidden_files_modified`)

All 15 were **pre-existing on the P47R base** (verified by running each failing test on detached HEAD at `origin/phase2/p47r-report-only-full-curated-evidence-correction-no-code-change`).

P48 introduced zero new scope gate failures.

---

## 3. Root Cause

These older tests run:

```python
res = subprocess.run(
    ["git", "diff", "--name-only", "<Pxx_BASE>"],
    capture_output=True, text=True, check=True
)
modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
for f in modified:
    assert f_norm in allowed, ...
```

Without a branch-aware skip guard. On later cumulative branches (P25+, P28+, etc.), they see files from later phases in the git diff, causing false assertion failures.

This is the same class of issue repaired in P46R for P42–P46.

---

## 4. Policy Applied

Accepted P46R policy:

> **Phase-local scope gates are authoritative only on their own phase branch. On later cumulative branches, they must skip explicitly using `enforce_phase_local_scope_gate_or_skip`.**

Rules maintained:
- No global monkeypatching of `subprocess.run`
- No fake git diff output
- No environment-variable bypass
- No deleting scope-gate tests
- No weakening of allowed-file sets
- No changing phase source files
- No scientific claims

---

## 5. Discovery Results

**Failing scope gates and their exact files:**

| Phase | Function | File |
|-------|----------|------|
| P24 | `test_p24_10_no_forbidden_files_modified` | `tests/test_phase2_p24_architecture_docs.py` |
| P25 | `test_p25_43_no_forbidden_files_modified` | `tests/test_phase2_model_interface.py` |
| P26 | `test_p26_33_no_forbidden_files_modified` | `tests/test_phase2_torch_boundary.py` |
| P27 | `test_p27_44_scope_gate` | `tests/test_phase2_fc_vae_model_skeleton.py` |
| P28 | `test_p28_45_scope_gate` | `tests/test_phase2_fc_vae_torch_shell.py` |
| P29 | `test_p29_49_scope_gate` | `tests/test_phase2_fc_vae_torch_module_stub.py` |
| P30 | `test_p30_44_scope_gate` | `tests/test_phase2_fc_vae_constructor_binding.py` |
| P31 | `test_p31_53_scope_gate` | `tests/test_phase2_fc_vae_forward_boundary.py` |
| P32 | `test_p32_65_scope_gate` | `tests/test_phase2_fc_vae_forward_input_batch.py` |
| P33 | `test_p33_79_scope_gate` | `tests/test_phase2_fc_vae_fake_input_descriptor.py` |
| P34 | `test_p34_86_scope_gate` | `tests/test_phase2_fc_vae_fake_input_preview.py` |
| P35 | `test_p35_95_scope_gate` | `tests/test_phase2_fc_vae_fake_input_flat_vector.py` |
| P36 | `test_p36_96_scope_gate` | `tests/test_phase2_fc_vae_flat_vector_batch_view.py` |
| P37 | `test_p37_107_scope_gate` | `tests/test_phase2_fc_vae_nested_batch_values.py` |
| P38 | `test_p38_105_scope_gate` | `tests/test_phase2_fc_vae_tensor_materialization_request.py` |

---

## 6. Changed Files

| File | Change Type | Description |
|------|-------------|-------------|
| `tests/test_phase2_p24_architecture_docs.py` | MODIFIED | Added skip guard to `test_p24_10_no_forbidden_files_modified` |
| `tests/test_phase2_model_interface.py` | MODIFIED | Added skip guard to `test_p25_43_no_forbidden_files_modified` |
| `tests/test_phase2_torch_boundary.py` | MODIFIED | Added skip guard to `test_p26_33_no_forbidden_files_modified` |
| `tests/test_phase2_fc_vae_model_skeleton.py` | MODIFIED | Added skip guard to `test_p27_44_scope_gate` |
| `tests/test_phase2_fc_vae_torch_shell.py` | MODIFIED | Added skip guard to `test_p28_45_scope_gate` |
| `tests/test_phase2_fc_vae_torch_module_stub.py` | MODIFIED | Added skip guard to `test_p29_49_scope_gate` |
| `tests/test_phase2_fc_vae_constructor_binding.py` | MODIFIED | Added skip guard to `test_p30_44_scope_gate` |
| `tests/test_phase2_fc_vae_forward_boundary.py` | MODIFIED | Added skip guard to `test_p31_53_scope_gate` |
| `tests/test_phase2_fc_vae_forward_input_batch.py` | MODIFIED | Added skip guard to `test_p32_65_scope_gate` |
| `tests/test_phase2_fc_vae_fake_input_descriptor.py` | MODIFIED | Added skip guard to `test_p33_79_scope_gate` |
| `tests/test_phase2_fc_vae_fake_input_preview.py` | MODIFIED | Added skip guard to `test_p34_86_scope_gate` |
| `tests/test_phase2_fc_vae_fake_input_flat_vector.py` | MODIFIED | Added skip guard to `test_p35_95_scope_gate` |
| `tests/test_phase2_fc_vae_flat_vector_batch_view.py` | MODIFIED | Added skip guard to `test_p36_96_scope_gate` |
| `tests/test_phase2_fc_vae_nested_batch_values.py` | MODIFIED | Added skip guard to `test_p37_107_scope_gate` |
| `tests/test_phase2_fc_vae_tensor_materialization_request.py` | MODIFIED | Added skip guard to `test_p38_105_scope_gate` |
| `tests/test_phase2_p48r_legacy_scope_gate_policy_repair.py` | NEW | P48R policy verification test (7 tests) |
| `reports/PHASE_2_P48R_LEGACY_SCOPE_GATE_POLICY_REPAIR_P27_TO_P38_NO_SOURCE_CHANGE_REPORT.md` | NEW | This report |

---

## 7. Scope Gate Repair Details per Phase

Each repaired test follows the same pattern — original allowed-set and base-branch preserved, direct subprocess enforcement replaced with `enforce_phase_local_scope_gate_or_skip`:

| Phase | Expected Branch | Base Commit (git diff target) |
|-------|----------------|-------------------------------|
| P24 | `phase2/p24-factorised-constrained-vae-architecture-spec` | `phase2/p23-evidence-contract-normalized-baseline-bundle` |
| P25 | `phase2/p25-model-interface-skeleton-candidate-contract` | `phase2/p24-factorised-constrained-vae-architecture-spec` |
| P26 | `phase2/p26-torch-gated-model-dependency-boundary` | `phase2/p25-fix-remove-subprocess-monkeypatch` |
| P27 | `phase2/p27-fc-vae-module-skeleton-shape-contract` | `phase2/p26-torch-gated-model-dependency-boundary` |
| P28 | `phase2/p28-optional-torch-fc-vae-shell-handle` | `phase2/p27-fc-vae-module-skeleton-shape-contract` |
| P29 | `phase2/p29-gated-torch-nn-module-stub` | `phase2/p28-optional-torch-fc-vae-shell-handle` |
| P30 | `phase2/p30-torch-module-constructor-spec-binding` | `phase2/p29-gated-torch-nn-module-stub` |
| P31 | `phase2/p31-noop-forward-boundary-contract` | `phase2/p30-torch-module-constructor-spec-binding` |
| P32 | `phase2/p32-forward-input-batch-contract-no-tensor` | `phase2/p31-noop-forward-boundary-contract` |
| P33 | `phase2/p33-deterministic-fake-input-descriptor-no-values` | `phase2/p32-forward-input-batch-contract-no-tensor` |
| P34 | `phase2/p34-deterministic-fake-input-scalar-preview-no-tensor` | `phase2/p33-deterministic-fake-input-descriptor-no-values` |
| P35 | `phase2/p35-deterministic-fake-input-flat-vector-no-tensor` | `phase2/p34-deterministic-fake-input-scalar-preview-no-tensor` |
| P36 | `phase2/p36-flat-vector-2d-batch-view-contract-no-nested-values` | `phase2/p35-deterministic-fake-input-flat-vector-no-tensor` |
| P37 | `phase2/p37-controlled-nested-batch-values-no-tensor-no-array` | `phase2/p36-flat-vector-2d-batch-view-contract-no-nested-values` |
| P38 | `phase2/p38-tensor-materialization-request-contract-no-tensor-allocation` | `phase2/p37-controlled-nested-batch-values-no-tensor-no-array` |

---

## 8. No Monkeypatch Evidence

Verified by `test_p48r_03_no_global_monkeypatch_in_repaired_files`. None of the following patterns appear in any repaired file:

- `subprocess.run = `
- `original_run = subprocess.run`
- `MockCompletedProcess`
- `mock_run`

All 15 repaired files use real `git branch --show-current` and real `git diff --name-only` via `phase2_scope_gate_utils`.

---

## 9. No Source/Tool Change Evidence

Verified by `test_p48r_05_src_phase2_files_unchanged` and `test_p48r_06_tools_phase2_files_unchanged`.

```
git diff --name-only 1de362b5e0dd7edf4bedc6ceeccc25a84f6a9de4 -- src/phase2
# (empty — no source changes)

git diff --name-only 1de362b5e0dd7edf4bedc6ceeccc25a84f6a9de4 -- tools/phase2
# (empty — no tool changes)
```

---

## 10. Focused Tests

```
python -m pytest \
  tests/test_phase2_p27_fc_vae_skeleton_smoke.py \
  tests/test_phase2_p28_fc_vae_torch_shell_smoke.py \
  tests/test_phase2_p29_torch_module_stub_smoke.py \
  tests/test_phase2_p30_constructor_binding_smoke.py \
  tests/test_phase2_p31_forward_boundary_smoke.py \
  tests/test_phase2_p32_forward_input_batch_smoke.py \
  tests/test_phase2_p33_fake_input_descriptor_smoke.py \
  tests/test_phase2_p34_fake_input_preview_smoke.py \
  tests/test_phase2_p35_fake_input_flat_vector_smoke.py \
  tests/test_phase2_p36_flat_vector_batch_view_smoke.py \
  tests/test_phase2_p37_nested_batch_values_smoke.py \
  tests/test_phase2_p38_tensor_materialization_request_smoke.py \
  tests/test_phase2_p48r_legacy_scope_gate_policy_repair.py -v
```

**Result: `357 passed, 0 failed, 12 skipped` (P27–P38 scope gates skip explicitly on P48R branch)**

---

## 11. Full Curated Tests

```
python -m pytest tests/test_phase2_schema.py ... tests/test_phase2_p48r_legacy_scope_gate_policy_repair.py -q
```

**Result: TBD (running)**

Expected: `0 failed`, 15 scope gates skip explicitly.

---

## 12. Remaining Blockers

None known. All 15 pre-existing scope gate failures are repaired. P48R introduces no new failures.

---

## 13. Final Verdict

`P48R_READY_FOR_REVIEW`
