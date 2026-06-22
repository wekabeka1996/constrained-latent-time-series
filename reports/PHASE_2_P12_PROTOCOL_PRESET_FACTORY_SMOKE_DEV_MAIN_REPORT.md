# Phase 2 P12: Protocol Preset Factory Smoke Dev Main Report

## 1. Task Summary
This task implements a narrow, explicit protocol preset factory for the Phase 2 dataset generation. The factory constructs `Phase2ArtifactRunRequest` objects for the approved presets (`smoke`, `dev`, and `main`), incorporating deterministic base seeds by family, custom split requirements, and validation rules. It does not perform any generation or execution of code.

## 2. Base Commit Verification
- Base branch: `phase2/p12-protocol-preset-factory-smoke-dev-main`
- Base commit: `12230a465c1d5d518a68263c036d80ddb8c1d0d6` (aligned P11 fix runner contract)

## 3. Files Created
- `src/phase2/protocol_presets.py`
- `tests/test_phase2_protocol_presets.py`
- `reports/PHASE_2_P12_PROTOCOL_PRESET_FACTORY_SMOKE_DEV_MAIN_REPORT.md`

## 4. Files Modified
- `src/phase2/__init__.py`

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 354 passed.
- **Compatibility Tests (Optional)**: 445 passed, 4 skipped.

## 8. Preset Factory Design Implemented
The factory consists of two dataclasses and three core helper/builder functions in `src/phase2/protocol_presets.py`:
- `ProtocolPresetName`: Enum defining `SMOKE`, `DEV`, and `MAIN` presets.
- `ProtocolPresetFactoryRequest`: Frozen dataclass containing parameters for generating the run request.
- `ProtocolPresetFactoryResult`: Frozen dataclass wrapping the resulting `Phase2ArtifactRunRequest` along with metrics and verification details.
- `validate_protocol_preset_factory_request`: Validates types, ranges, duplicates, missing templates, and paths.
- `build_split_request_for_preset`: Maps each of the 5 splits (`smoke`, `zero_shot_train`, `zero_shot_eval`, `fewshot_train`, `fewshot_eval`) to a specific `SplitArtifactRequest`.
- `build_phase2_preset_run_request`: Conforming to P11 contract, sequences split requests in the exact required order.
- `build_phase2_preset_factory_result`: Summarizes sample counts, splits, and compiles the final factory result object.

## 9. Public Constants
The following public constants are explicitly exposed in `src/phase2/protocol_presets.py` and re-exported in `src/phase2/__init__.py`:
- `APPROVED_PHASE2_PRESET_SAMPLE_COUNTS`: Encodes preset sample counts.
- `APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY`: Base seeds (AR=12001, ARMA=12001, GARCH=12002, ARMA_GARCH=12003).
- `APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT`: Subdirectory names.
- `APPROVED_PHASE2_SAMPLES_FILENAME`: `"samples.jsonl"`
- `APPROVED_PHASE2_MANIFEST_FILENAME`: `"manifest.json"`

## 10. Smoke/Dev/Main Count Behavior
Counts per active family are determined explicitly via `get_preset_sample_count_by_family(preset_name)`:
- `SMOKE`: 1,000 samples per active family.
- `DEV`: 10,000 samples per active family.
- `MAIN`: 100,000 samples per active family.

## 11. Split Construction Behavior
For each preset, split requests are constructed in the exact sequence:
1. `smoke`: AR, ARMA, GARCH, ARMA_GARCH at full preset count.
2. `zero_shot_train`: AR, ARMA, GARCH at preset count; ARMA_GARCH is omitted/excluded (count 0).
3. `zero_shot_eval`: ARMA_GARCH at preset count (C-only).
4. `fewshot_train`: ARMA_GARCH at preset count (C-only).
5. `fewshot_eval`: ARMA_GARCH at preset count (C-only).

## 12. Zero-shot C Holdout Behavior
For `zero_shot_train`, `enforce_zero_shot_c_train_exclusion` is set to `True`, and the `ARMA_GARCH` family is excluded from the split request's sample counts. This is validated by split runner guards.

## 13. Fewshot Limitation Note
No 1% or 5% fewshot evaluation slicing semantics are claimed or implemented in P12. The fewshot splits are currently C-only placeholders using explicit preset counts. Slicing logic is reserved for Phase 2 P13.

## 14. No-Execution Guarantee
The factory implemented in `src/phase2/protocol_presets.py` strictly operates as a request builder:
- It does **not** call `run_phase2_artifact_generation`.
- It does **not** call `write_dataset_artifacts`.
- It does **not** call `build_dataset_in_memory`.
- It does **not** write any files or create directories on disk.
- It only returns `Phase2ArtifactRunRequest` and `ProtocolPresetFactoryResult`.

## 15. Scope Confirmation
The implementation adheres to the narrow scope:
- No dependencies on `torch`, `numpy`, `pandas`, `yaml`, or `argparse` were introduced or imported.
- No model, training, or metric scoring code was written or imported.
- Unit tests in `tests/test_phase2_protocol_presets.py` do not create any directories or write any files.

## 16. Remaining Blockers
None. All tests pass, and contracts are fully aligned.

## 17. Post-Commit/Push Evidence
Evidence is generated and logged upon committing and pushing.

## 18. Final Verdict
**PASS**. The protocol preset factory successfully constructs Phase 2 run requests under all constraints.
