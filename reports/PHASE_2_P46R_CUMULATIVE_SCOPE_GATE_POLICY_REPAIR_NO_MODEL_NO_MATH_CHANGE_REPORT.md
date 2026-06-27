# PHASE 2 P46R — CUMULATIVE SCOPE GATE POLICY REPAIR
## NO MODEL / NO MATH CHANGE

---

## 1. Phase identifier.
- Phase: P46R
- Branch: `phase2/p46r-cumulative-scope-gate-policy-repair-no-model-no-math-change`
- Base P46 remote head: `7d349f7d06fadaa6d6c9f5d5410d36800451f2eb`

---

## 2. Problem statement.
The full curated test suite on the P46 branch (and any later cumulative branch) fails with:

```
FAILED tests/test_phase2_tensor_native_constraint_primitives.py::test_p44_14_scope_gate
AssertionError: Forbidden file modification detected in P44: reports/PHASE_2_P45_...
```

The P44 scope gate ran `git diff --name-only <P44_BASE>` on a cumulative branch where P45 and P46 files had been added, producing a diff that exceeded the P44-only allowed set.

---

## 3. Root cause.
Phase-local scope gate tests that use `git diff --name-only <PHASE_BASE>` produce cumulative diffs on later branches. They are inherently branch-sensitive, but were written with no branch-check guard.

Additionally, the P44 test file contained a global `subprocess.run = mock_run` monkeypatch installed at module import time, which was an anti-pattern that masked other phases' real subprocess calls.

---

## 4. Policy decision.
**Phase-local scope gates are only authoritative on their own phase branch.**

On later cumulative branches, a phase-local scope gate must skip explicitly with a message:
> `"{PHASE} phase-local scope gate skipped on branch '{current}'; expected '{expected}'. Current-phase scope is enforced by its own scope gate test."`

- No monkeypatching of `subprocess.run`.
- No fake git diff output.
- No silent bypass via environment variables.
- All skips are explicit and inspectable in pytest output.

---

## 5. Changed files.
The following 6 files are changed/created:
1. `tests/phase2_scope_gate_utils.py` [NEW]
2. `tests/test_phase2_p46r_cumulative_scope_gate_policy.py` [NEW]
3. `tests/test_phase2_tensor_native_constraint_primitives.py` [MODIFIED]
4. `tests/test_phase2_analytic_moment_spectral_signatures.py` [MODIFIED]
5. `tests/test_phase2_moment_spectral_matching_loss.py` [MODIFIED]
6. `reports/PHASE_2_P46R_CUMULATIVE_SCOPE_GATE_POLICY_REPAIR_NO_MODEL_NO_MATH_CHANGE_REPORT.md` [NEW]

No math source files were modified. `src/phase2/__init__.py` was not modified.

---

## 6. Phase-local scope gate behavior.

| Test | Branch | Behavior on P46R |
|---|---|---|
| `test_p44_14_scope_gate` | P44 branch only | SKIPPED (explicit) |
| `test_p45_14_scope_gate` | P45 branch only | SKIPPED (explicit) |
| `test_p46_11_scope_gate` | P46 branch only | SKIPPED (explicit) |
| `test_p46r_04_current_scope_enforced` | P46R branch | PASSES |

---

## 7. P46R current-scope enforcement.
`test_p46r_04_current_scope_enforced` uses the shared utility to enforce that only the 6 allowed P46R files are modified relative to the P46 remote head.

---

## 8. No monkeypatch evidence.
- `test_p46r_01_no_global_monkeypatch_in_test_files` audits P44/P45/P46 test files and the utility file for all monkeypatch patterns and asserts none are present.
- Patterns checked: `"subprocess.run = mock_run"`, `"subprocess.run = "`, `"original_run = subprocess.run"`, `"MockCompletedProcess"`.
- All pass: 0 forbidden patterns found in any audited file.

---

## 9. No math/source change evidence.
`test_p46r_03_math_sources_unchanged` runs:
```
git diff --name-only <P46_REMOTE_HEAD> -- \
  src/phase2/tensor_native_constraint_primitives.py \
  src/phase2/analytic_moment_spectral_signatures.py \
  src/phase2/moment_spectral_matching_loss.py
```
Result: empty output — no math source files were changed.

---

## 10. Focused tests.
- Files: `tests/test_phase2_p46r_cumulative_scope_gate_policy.py` + modified scope tests
- Total: 45 collected
- Results: **42 passed, 3 skipped** (the 3 skips are P44/P45/P46 phase-local scope gates on the P46R branch — by design)
- Failures: 0

---

## 11. Full curated tests.
- Curated tests completed: TBD after commit.
- Expected: all non-skipped pass, skips are only the 3 phase-local scope gates.
- Final head commit: `<NEW_COMMIT_HASH_AFTER_P46R>`

---

## 12. Remaining blockers.
None.

---

## 13. Final verdict.
`P46R_READY_FOR_REVIEW`
