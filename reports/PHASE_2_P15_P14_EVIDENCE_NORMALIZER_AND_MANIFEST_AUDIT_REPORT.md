# Phase 2 P15: P14 Evidence Normalizer and Manifest Audit Report

## 1. Goal
Implement a reusable manifest/hash audit utility to verify Phase 2 generated artifacts, normalize all output paths to be repository-relative to prevent local machine path leakage, and audit the dry-run artifacts without introducing new dependencies or violating scope.

## 2. Base Commit Verification
- Base branch: `phase2/p14-smoke-artifact-generation-dry-run`
- Base commit: `a61fdf591a5a51c925e3b599ab7f147008bd5f92`

## 3. Files Created
- `tools/phase2/audit_phase2_artifact_manifest.py`
- `tests/test_phase2_p15_manifest_audit.py`
- `reports/PHASE_2_P15_P14_EVIDENCE_NORMALIZER_AND_MANIFEST_AUDIT_REPORT.md`

## 4. Files Modified
- `tools/phase2/run_p14_smoke_artifact_dry_run.py`
- `tests/test_phase2_p14_smoke_dry_run.py`
- `reports/PHASE_2_P14_SMOKE_ARTIFACT_GENERATION_DRY_RUN_REPORT.md`

## 5. Audit Utility Public API Specification
The following public functions were implemented in `tools/phase2/audit_phase2_artifact_manifest.py`:
1. `repository_root() -> pathlib.Path`: Resolves the absolute path to the repository root.
2. `normalize_repo_relative_path(path, repo_root) -> str`: Normalizes paths under `repo_root` to be POSIX repo-relative. Rejects empty paths, paths outside `repo_root`, and any leaked absolute prefixes (e.g. `C:/`, `file:///`, etc.).
3. `sha256_file(path) -> str`: Wrapper around `src.phase2.artifacts.sha256_file`.
4. `audit_manifest_file(manifest_path, repo_root) -> dict`: Parses `manifest.json`, validates it has correct `artifact_type`, checks `samples_sha256` matches the physical `samples.jsonl` file, and returns path-normalized details.
5. `audit_phase2_artifact_root(output_root_dir, expected_subdirs, expected_counts_by_subdir, repo_root) -> dict`: Validates exactly 6 expected split directories, verifies each directory contains only `samples.jsonl` and `manifest.json`, verifies split counts, zst C-train exclusion, C-family exclusivity in evaluation/fewshot splits, and returns a path-normalized summary.
6. `audit_phase2_artifacts(output_root_dir: str) -> dict`: Convenience wrapper for auditing P14 SMOKE expected layout.
7. `compact_json(data: dict) -> str`: Serializes a dictionary to a sorted compact JSON string.
8. `assert_no_absolute_paths_in_data(data) -> None`: Recursively validates that no absolute path strings (Windows drives, Linux home/user paths, or file URIs) exist in the data.

## 6. Execution & Verification Results
When `python tools/phase2/run_p14_smoke_artifact_dry_run.py` is executed, the following sanitized, path-normalized compact JSON is printed to stdout:

```json
{"fewshot_seed_plan_verified":true,"fewshot_superset_note":"sample_id prefix not required because split/artifact identity differs; seed prefix semantics verified instead.","output_root_dir":"phase2_artifacts/p14_smoke_dry_run","per_split":[{"artifact_subdir":"smoke","manifest_path":"phase2_artifacts/p14_smoke_dry_run/smoke/manifest.json","manifest_sha256":"45bb306545a7eac8b97c18a9390289db261f32d270e4ed76265c33f69f039e0f","sample_count":4000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/smoke/samples.jsonl","samples_sha256":"8cdbbec022abf7c2d557069f4b3d4acc2038ce7738e7e259be44629bfe3cc652","split_name":"smoke"},{"artifact_subdir":"zero_shot_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_train/manifest.json","manifest_sha256":"5a09576207f3a05e2a0ab47288d13fc1dde740d422754452241ded9f61bd9abc","sample_count":3000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_train/samples.jsonl","samples_sha256":"4732e83d0a90a2f162287e7b854765310f36442f13ee71b993e3b3c400b0cdbb","split_name":"zero_shot_train"},{"artifact_subdir":"zero_shot_eval","manifest_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_eval/manifest.json","manifest_sha256":"a987e4550c809a655f10b4e84d0c8f8fc8647c52bf2a04fc0681db34ab0cb388","sample_count":1000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_eval/samples.jsonl","samples_sha256":"172351896ce7e44a2d2231c6b8275da1d7e184eff7a972f7c42ab31b8b7c0901","split_name":"zero_shot_eval"},{"artifact_subdir":"fewshot_1pct_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_1pct_train/manifest.json","manifest_sha256":"6ed9187e26afe05e383fa18d7bb3ed13bc64b0194c8c5724ed8539129fb73422","sample_count":10,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_1pct_train/samples.jsonl","samples_sha256":"0b922b408435bdccd676e9c24b75d7edb509e6cbde9c247f336db133e9d1a4e0","split_name":"fewshot_train"},{"artifact_subdir":"fewshot_5pct_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_5pct_train/manifest.json","manifest_sha256":"106e9839525ea10e7f829d42eedf065ce2c43f45cc0399fe92cb6a52bf8ac122","sample_count":50,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_5pct_train/samples.jsonl","samples_sha256":"50ab9e5b566602bd74dbaa6913634fa83943bfa344d0e7466c228c8cf7cef01e","split_name":"fewshot_train"},{"artifact_subdir":"fewshot_eval","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_eval/manifest.json","manifest_sha256":"ad53aad00fb3f3d5b359ae4eaa8faf64cac0cbe2bf32109bf1606c1c5c1b9ac2","sample_count":1000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_eval/samples.jsonl","samples_sha256":"84e25a7d8f958cda2167b07fc043d6591494c02907845d3b552dcb7ce3e5abcc","split_name":"fewshot_eval"}],"reason":"phase2_p14_smoke_dry_run_validation_complete","total_sample_count":9060,"total_split_count":6,"unexpected_files":[],"verdict":"PASS","zero_shot_c_train_count":0}
```

## 7. Tests Run and Exact Results
The following test suites were run and passed successfully:
- **Phase 2 Tests (Required)**: `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py -q`
  - Result: **379 passed** in 67.41 seconds.
- **Compatibility Tests (Optional)**: `python -m pytest tests/test_config.py ...`
  - Result: **470 passed, 4 skipped** in 66.16 seconds.

## 8. No Forbidden Dependencies / Code Style
- Verified that `tools/phase2/audit_phase2_artifact_manifest.py` and `tools/phase2/run_p14_smoke_artifact_dry_run.py` contain no imports of `torch`, `numpy`, `pandas`, `yaml`, or `argparse`.
- Verified that they do not implement any model code, training loops, or evaluation metrics.

## 9. Verdict
**P15_READY_FOR_REVIEW**
