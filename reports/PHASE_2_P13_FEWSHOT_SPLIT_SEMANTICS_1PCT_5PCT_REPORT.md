# Phase 2 P13: Fewshot Split Semantics 1% / 5% / Superset Report

## 1. Task Summary
This task implements explicit deterministic 1% and 5% few-shot C-train split semantics, superset validations, and seed range non-overlap validations for the Phase 2 dataset presets. P13 replaces the temporary placeholders from P12 with correct derived base seeds and validation constraints.

## 2. Base Commit Verification
- Base branch: `phase2/p12-protocol-preset-factory-smoke-dev-main`
- Base commit: `526b6ef0bf6a94f850d71f9f97e3b7f7588c1f43` (which contains the accepted P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6`)

## 3. Files Created
- `reports/PHASE_2_P13_FEWSHOT_SPLIT_SEMANTICS_1PCT_5PCT_REPORT.md`

## 4. Files Modified
- `src/phase2/protocol_presets.py`
- `src/phase2/__init__.py`
- `tests/test_phase2_protocol_presets.py`
- `reports/PHASE_2_P12_PROTOCOL_PRESET_FACTORY_SMOKE_DEV_MAIN_REPORT.md`

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 351 passed.
- **Compatibility Tests (Optional)**: 442 passed, 4 skipped.

## 8. P13 Fewshot Semantics Implemented
P13 derived base seeds from P3 roots by applying a public stride of `10_000_000` to prevent direct overlap between the fewshot train and the evaluation holdouts for large preset sizes.
- Root Seeds:
  - `zero_shot_eval`: `12003`
  - `fewshot_train`: `12101`
  - `fewshot_eval`: `12105`
- Derived Base Seeds:
  - `zero_shot_eval`: `12003`
  - `fewshot_train` (`fewshot_1pct_train` & `fewshot_5pct_train`): `10_012_101`
  - `fewshot_eval`: `20_012_105`

## 9. 1% / 5% Count Table
| Preset | Fewshot 1% Count | Fewshot 5% Count |
| :--- | :--- | :--- |
| **SMOKE** | 10 | 50 |
| **DEV** | 100 | 500 |
| **MAIN** | 1,000 | 5,000 |

## 10. Superset Proof
Fewshot 5% is a deterministic prefix superset of Fewshot 1% because:
1. Both splits share the exact same ARMA_GARCH base seed `10_012_101`.
2. The fewshot 1% count ($N_1$) is strictly less than the fewshot 5% count ($N_5$).
3. The sample generation seed sequence is $S_i = \text{base\_seed} + i$ for $i \in [0, N-1]$.
4. The seed range for 1% is $[10012101, 10012101 + N_1 - 1]$ which is a prefix subset of the 5% seed range $[10012101, 10012101 + N_5 - 1]$.
5. The simulation seed sequence is $S_{sim, i} = S_i + 1,000,000$, ensuring the simulation seeds also form a deterministic prefix superset.

## 11. C Eval Holdout Non-Overlap Proof
The C evaluation holdout seed ranges do not overlap fewshot train seed ranges. Let $N_{eval}$ be the preset evaluation count:
- SMOKE: $N_{eval} = 1000$
  - `fewshot_5pct_train` range: $[10_012_101, 10_012_150]$
  - `zero_shot_eval` range: $[12003, 13002]$
  - `fewshot_eval` range: $[20_012_105, 20_013_104]$
  - **No overlaps** exist in generation or simulation ranges.
- DEV: $N_{eval} = 10000$
  - `fewshot_5pct_train` range: $[10_012_101, 10_012_600]$
  - `zero_shot_eval` range: $[12003, 22002]$
  - `fewshot_eval` range: $[20_012_105, 20_022_104]$
  - **No overlaps** exist in generation or simulation ranges.
- MAIN: $N_{eval} = 100000$
  - `fewshot_5pct_train` range: $[10_012_101, 10_017_100]$
  - `zero_shot_eval` range: $[12003, 112002]$
  - `fewshot_eval` range: $[20_012_105, 20_112_104]$
  - **No overlaps** exist in generation or simulation ranges.

All non-overlaps are validated programmatically via `validate_p13_fewshot_seed_plan_for_preset`.

## 12. Split Order
Run requests sequence exactly 6 splits in the order:
1. `smoke` (subdir: `smoke`, SplitName: `SMOKE`)
2. `zero_shot_train` (subdir: `zero_shot_train`, SplitName: `ZERO_SHOT_TRAIN`)
3. `zero_shot_eval` (subdir: `zero_shot_eval`, SplitName: `ZERO_SHOT_EVAL`)
4. `fewshot_1pct_train` (subdir: `fewshot_1pct_train`, SplitName: `FEWSHOT_TRAIN`)
5. `fewshot_5pct_train` (subdir: `fewshot_5pct_train`, SplitName: `FEWSHOT_TRAIN`)
6. `fewshot_eval` (subdir: `fewshot_eval`, SplitName: `FEWSHOT_EVAL`)

## 13. Zero-shot C Exclusion Behavior
For `zero_shot_train`, the ARMA_GARCH family is omitted, and the request is built with `enforce_zero_shot_c_train_exclusion=True`.

## 14. No-Execution Guarantee
The implementation does **not** execute any artifact generation, dataset building, or writing:
- P13 does not execute generation.
- P13 does not write artifacts.
- P13 does not call P11 runner (`run_phase2_artifact_generation`).
- P13 does not call P10 writer (`write_dataset_artifacts`).
- P13 does not call P9 dataset builder (`build_dataset_in_memory`).

## 15. Scope Confirmation
- P13 does not implement metrics.
- P13 does not implement training.
- P13 does not implement model code.
- P13 does not implement CLI.
- P13 does not implement config loader.
- P13 does not introduce `torch`/`numpy`/`pandas`/`yaml`/`argparse` dependencies.
- All code in `src/phase2/protocol_presets.py` is written in standard library Python only.
- Test suite in `tests/test_phase2_protocol_presets.py` does not create any files or directories on disk.

## 16. Remaining Blockers
None.

## 17. Post-Commit/Push Evidence
- Branch: `phase2/p13-fewshot-split-semantics-1pct-5pct`
- Commit: `276f18713a6ba42a09ce57e34cab4600627346bb`

## 18. Final Verdict
**P13_READY_FOR_REVIEW**
