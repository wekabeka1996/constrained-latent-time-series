# PHASE 2 P11 FIX: Runner Contract Alignment Report

## 1. Task Summary
This task fixes the Phase 2 P11 Split Runner implementation to match the approved P11 runner contract exactly. We corrected the structures of `SplitArtifactRequest`, `Phase2ArtifactRunRequest`, `SplitArtifactRunResult`, and `Phase2ArtifactRunResult` to match the exact dataclass fields, moved configuration options from run-level to split-level requests where required, and implemented output target pair validation to reject duplicate path destinations.

- All file serialization is handled exclusively by P10 `write_dataset_artifacts`.
- No models, training loops, argparse, yaml, configs, or statistical metrics were imported or implemented.

## 2. Base Commit Verification
- **Branch**: `phase2/p11-fix-runner-contract-alignment`
- **Base Commit HEAD**: `8ca56725da4ecd44e663666875608276e8410d3a` (child of `3d2b64881da94dbea9abf7e76750d3b18a65d4cf` which was verified as an ancestor).

## 3. Files Created
- `reports/PHASE_2_P11_FIX_RUNNER_CONTRACT_ALIGNMENT_REPORT.md`: This report.

## 4. Files Modified
- `src/phase2/split_runner.py`: Updated fields of all dataclasses, added output target pair checks, updated mappings and helper functions, and updated the runner to compute aggregate metrics.
- `tests/test_phase2_split_runner.py`: Updated split and run request constructors to use the updated schemas and added tests for output target checks, duplicate name allowances, and updated helper mapping assertions.
- `reports/PHASE_2_P11_ARTIFACT_BACKED_SPLIT_GENERATION_RUNNER_REPORT.md`: Updated to match the revised schemas.

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Contract Fixes Made
1. **`SplitArtifactRequest`**:
   - Added: `simulation_template`, `artifact_subdir`, `enforce_zero_shot_c_train_exclusion`, `sample_id_hash_len`.
   - Removed all default values/factories.
2. **`Phase2ArtifactRunRequest`**:
   - Replaced: `artifact_dir` / `artifact_subdir` / `splits` / global simulation / leakage / hash parameters with `output_root_dir` and `split_requests`.
   - Removed all default values/factories.
3. **`SplitArtifactRunResult`**:
   - Added: `artifact_subdir`, `dataset_total_count`, `zero_shot_c_train_count`, `sample_ids`, `reason`.
   - Removed all default values/factories.
4. **`Phase2ArtifactRunResult`**:
   - Added: `output_root_dir`, `total_sample_count`, `total_split_count`, `reason`.
   - Removed all default values/factories.
5. **Helper Functions**:
   - `build_dataset_request_for_split` maps `protocol_name` from run request, and the rest from the split request.
   - `build_artifact_write_request_for_split` constructs `output_dir = output_root_dir / split_request.artifact_subdir`.

## 7. Zero-Shot Leakage Behavior
Zero-shot C Train leakage checking is executed per split. If a split request has `split_name == SplitName.ZERO_SHOT_TRAIN` and `enforce_zero_shot_c_train_exclusion == True`, any `FamilyId.ARMA_GARCH` count `> 0` will cause validation to fail-fast with a `ValueError`.

## 8. Per-Split `artifact_subdir` Behavior
Each split request specifies its own `artifact_subdir`. Run request validation checks that `artifact_subdir` is unique across all split requests.

## 9. Output Target Pair Guard
Validation ensures that no two split requests map to the same target output file paths. If two split requests share both `artifact_subdir` and `samples_filename`, or share both `artifact_subdir` and `manifest_filename`, it is rejected.
However, duplicate `split_name` across different subdirectories is explicitly allowed.

## 10. Determinism Evidence
All generated artifacts are written deterministically via P10. Runner preserves request split order exactly in the output result list.

## 11. Scope Confirmation
No third-party packages such as `torch`, `numpy`, `pandas`, `yaml`, or `argparse` are imported or implemented in `split_runner.py`.

## 12. Commands Run and Results
- Required suite:
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py -q`
  - Result: **305 passed**
- Compatibility suite:
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py -q`
  - Result: **396 passed, 4 skipped**

## 13. Post-Push Evidence
- **Branch Name**: `phase2/p11-fix-runner-contract-alignment`
- **Commit Hash**: [Pending stage and commit]
- **Remote Branch Check**: [Pending push]

## 14. Final Verdict
P11_FIX_READY_FOR_REVIEW
