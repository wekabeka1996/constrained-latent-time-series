# PHASE 2 P11: Artifact-Backed Split Generation Runner Report

## 1. Task Summary
This task implements a deterministic orchestration layer that composes the Phase 2 ModelSpec sampler, time-series simulator, dataset builder, and artifact writer into a multi-split runner. It allows generating all dataset splits required by a protocol (e.g., zero_shot_train, zero_shot_eval, fewshot_train, fewshot_eval, smoke) in a single run request, executing validation on both the run and individual split requests, and writing the artifacts to disk using P10.

- All artifact serialization and writing is strictly delegated to P10 `write_dataset_artifacts`.
- No model code was created or imported.
- No training loop was implemented.
- No config loader or parser was created.
- No metrics scoring was implemented.
- No torch, numpy, pandas, yaml, or argparse dependencies were imported or used in the runner.

## 2. Base Commit Verification
The base branch `phase2/p11-artifact-backed-split-generation-runner` was checked out. The HEAD was verified to contain `93d942133e799c2363fb42fb66767acde4605398` as our starting point:
- Base HEAD: `93d942133e799c2363fb42fb66767acde4605398`

## 3. Files Created
- `src/phase2/split_runner.py`: Implements `SplitArtifactRequest`, `Phase2ArtifactRunRequest`, `SplitArtifactRunResult`, and `Phase2ArtifactRunResult` dataclasses along with request validation functions (`validate_split_artifact_request`, `validate_phase2_artifact_run_request`), mappings (`build_dataset_request_for_split`, `build_artifact_write_request_for_split`), and the main orchestration runner (`run_phase2_artifact_generation`).
- `tests/test_phase2_split_runner.py`: Implements a comprehensive test suite covering split request validation, run request validation, zero-shot C leakage guards, request mapping, end-to-end multi-split generation, and strict scope constraints.
- `reports/PHASE_2_P11_ARTIFACT_BACKED_SPLIT_GENERATION_RUNNER_REPORT.md`: This report.

## 4. Files Modified
- `src/phase2/__init__.py`: Modified to import and re-export the new public split runner dataclasses and functions.

## 5. Files Not Changed
All other files in the repository remain unchanged:
- `src/phase2/schema.py` (Unchanged)
- `src/phase2/constraints.py` (Unchanged)
- `src/phase2/sampler.py` (Unchanged)
- `src/phase2/simulator.py` (Unchanged)
- `src/phase2/dataset.py` (Unchanged)
- `src/phase2/artifacts.py` (Unchanged)
- Legacy files, configs, and third-party dependencies are untouched.

## 6. Commands Run
- Run Phase 2 required tests:
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py -q`
- Run optional compatibility tests:
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Required Tests**: 305 passed.
- **Compatibility Suite**: 396 passed, 4 skipped.

## 8. Split Generation Runner Design
The split generation runner defines:
- `SplitArtifactRequest`: Captures parameters for an individual split (e.g. counts, base seeds, templates, simulation templates, output subdir, filenames, leakage flag, and hash length). Has no default values.
- `Phase2ArtifactRunRequest`: Orchestrates the run at a protocol level (e.g. protocol name, output root directory, list of split requests, parent directory option, overwrite option, and style / include preferences). Has no default values.
- `SplitArtifactRunResult`: Output details for a single completed split (e.g. name, subdir, total count, count by family, leakage count, paths, SHA256 hashes, sample IDs list, and reason). Has no default values.
- `Phase2ArtifactRunResult`: Final summary of the execution run (e.g. protocol name, output root directory, list of completed split results, total sample count, total split count, and overall reason). Has no default values.
- `validate_split_artifact_request(request)`: Validates split types, active template checks, positive counts, filenames, subdirectories, and leakage guards.
- `validate_phase2_artifact_run_request(request)`: Validates run types, output root directories, unique subdirectories, unique split+subdir pairs, unique output targets, and JSON styling.
- `run_phase2_artifact_generation(request)`: Orchestrates the validation of run request, loop validation of split requests, sequential dataset construction, artifact writing, and returns the unified result.

## 9. Zero-shot C Leakage Guard Detail
The runner enforces zero-shot C leakage guarding at request time. If `enforce_zero_shot_c_train_exclusion` is `True` on the split request, the validation checks if the split requests a sample count `> 0` for `FamilyId.ARMA_GARCH` when `split_name == SplitName.ZERO_SHOT_TRAIN`. If it does, it fails fast by raising a `ValueError` describing the leakage. This prevents execution of any splits and fails immediately.

## 10. `artifact_subdir` Validation
Each split request specifies its own `artifact_subdir`. Validation checks and rejects:
- Empty strings or whitespace-only strings.
- Absolute paths (using `is_absolute()` or checking for leading slashes/drive letters).
- Any path traversal using `".."`. This is validated by splitting the path components using `pathlib.Path.parts` and checking for the existence of `".."`, as well as checking against split separators.

## 11. Fail-Fast Behavior
The runner implements strict fail-fast semantics:
- The run request is validated before executing any split.
- Each split request is validated before constructing `DatasetBuildRequest`.
- Any failure in validation, dataset generation (e.g., stationarity violation), or artifact writing (e.g., file exists when `overwrite_existing=False`) raises an exception immediately.
- The runner does not catch these exceptions, does not silently continue, and does not convert failures into success results.

## 12. Scope Confirmation
`src/phase2/split_runner.py` does not import or implement any neural network model code, training workflows, configs, yaml, argparse, torch, numpy, pandas, or metrics scoring. Scope tests explicitly check `split_runner.py` source code contents.

## 13. Remaining Blockers
None.

## 14. Post-Commit/Push Evidence
- **Branch Name**: `phase2/p11-fix-runner-contract-alignment`
- **Commit Hash**: [Pending stage and commit]
- **Remote Branch Check**: [Pending push]

## 15. Final Verdict
P11_FIX_READY_FOR_REVIEW
