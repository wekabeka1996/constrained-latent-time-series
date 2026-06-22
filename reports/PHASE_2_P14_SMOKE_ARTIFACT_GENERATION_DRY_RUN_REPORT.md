# Phase 2 P14: Smoke Artifact Generation Dry-Run Report

> [!NOTE]
> P14 report normalized by P15 to remove local absolute paths.

## 1. Task Summary
This task executes a controlled dry-run of the Phase 2 artifact generation pipeline using the SMOKE preset, sequencing through 6 splits. The outputs are verified to ensure deterministic generation, correct split counts, zero-shot C exclusion, prefix superset seed semantics, manifest/sample SHA256 integrity, and no unexpected files.

## 2. Base Commit Verification
- Base branch: `phase2/p13-fewshot-split-semantics-1pct-5pct`
- Base commit: `ccc79de23784be86ead0de3a9bc146c7e8d5f7f4` (re-verifying ancestor P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6`)

## 3. Files Created
- `tools/phase2/run_p14_smoke_artifact_dry_run.py`
- `tests/test_phase2_p14_smoke_dry_run.py`
- `reports/PHASE_2_P14_SMOKE_ARTIFACT_GENERATION_DRY_RUN_REPORT.md`

## 4. Files Modified
- `.gitignore` (added `phase2_artifacts/`)

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `$env:PYTHONPATH="."; python tools/phase2/run_p14_smoke_artifact_dry_run.py`
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 367 passed.
- **Compatibility Tests (Optional)**: 458 passed, 4 skipped.

## 8. Dry-Run Command Executed
```bash
$env:PYTHONPATH="."; python tools/phase2/run_p14_smoke_artifact_dry_run.py
```

Stdout JSON result:
```json
{"fewshot_seed_plan_verified":true,"fewshot_superset_note":"sample_id prefix not required because split/artifact identity differs; seed prefix semantics verified instead.","output_root_dir":"phase2_artifacts/p14_smoke_dry_run","per_split":[{"artifact_subdir":"smoke","manifest_path":"phase2_artifacts/p14_smoke_dry_run/smoke/manifest.json","manifest_sha256":"45bb306545a7eac8b97c18a9390289db261f32d270e4ed76265c33f69f039e0f","sample_count":4000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/smoke/samples.jsonl","samples_sha256":"8cdbbec022abf7c2d557069f4b3d4acc2038ce7738e7e259be44629bfe3cc652","split_name":"smoke"},{"artifact_subdir":"zero_shot_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_train/manifest.json","manifest_sha256":"5a09576207f3a05e2a0ab47288d13fc1dde740d422754452241ded9f61bd9abc","sample_count":3000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_train/samples.jsonl","samples_sha256":"4732e83d0a90a2f162287e7b854765310f36442f13ee71b993e3b3c400b0cdbb","split_name":"zero_shot_train"},{"artifact_subdir":"zero_shot_eval","manifest_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_eval/manifest.json","manifest_sha256":"a987e4550c809a655f10b4e84d0c8f8fc8647c52bf2a04fc0681db34ab0cb388","sample_count":1000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/zero_shot_eval/samples.jsonl","samples_sha256":"172351896ce7e44a2d2231c6b8275da1d7e184eff7a972f7c42ab31b8b7c0901","split_name":"zero_shot_eval"},{"artifact_subdir":"fewshot_1pct_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_1pct_train/manifest.json","manifest_sha256":"6ed9187e26afe05e383fa18d7bb3ed13bc64b0194c8c5724ed8539129fb73422","sample_count":10,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_1pct_train/samples.jsonl","samples_sha256":"0b922b408435bdccd676e9c24b75d7edb509e6cbde9c247f336db133e9d1a4e0","split_name":"fewshot_train"},{"artifact_subdir":"fewshot_5pct_train","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_5pct_train/manifest.json","manifest_sha256":"106e9839525ea10e7f829d42eedf065ce2c43f45cc0399fe92cb6a52bf8ac122","sample_count":50,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_5pct_train/samples.jsonl","samples_sha256":"50ab9e5b566602bd74dbaa6913634fa83943bfa344d0e7466c228c8cf7cef01e","split_name":"fewshot_train"},{"artifact_subdir":"fewshot_eval","manifest_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_eval/manifest.json","manifest_sha256":"ad53aad00fb3f3d5b359ae4eaa8faf64cac0cbe2bf32109bf1606c1c5c1b9ac2","sample_count":1000,"samples_path":"phase2_artifacts/p14_smoke_dry_run/fewshot_eval/samples.jsonl","samples_sha256":"84e25a7d8f958cda2167b07fc043d6591494c02907845d3b552dcb7ce3e5abcc","split_name":"fewshot_eval"}],"reason":"phase2_p14_smoke_dry_run_validation_complete","total_sample_count":9060,"total_split_count":6,"unexpected_files":[],"verdict":"PASS","zero_shot_c_train_count":0}
```

## 9. Output Root Path
- `phase2_artifacts/p14_smoke_dry_run`

## 10. Generated Files Summary
| Split | Subdirectory | Samples Filename | Manifest Filename | Samples Size (Bytes) | Manifest Size (Bytes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **smoke** | `smoke` | `samples.jsonl` | `manifest.json` | 16,180,061 | 221,772 |
| **zero_shot_train** | `zero_shot_train` | `samples.jsonl` | `manifest.json` | 11,667,123 | 191,713 |
| **zero_shot_eval** | `zero_shot_eval` | `samples.jsonl` | `manifest.json` | 4,629,938 | 69,602 |
| **fewshot_1pct_train** | `fewshot_1pct_train` | `samples.jsonl` | `manifest.json` | 46,371 | 1,315 |
| **fewshot_5pct_train** | `fewshot_5pct_train` | `samples.jsonl` | `manifest.json` | 231,956 | 4,195 |
| **fewshot_eval** | `fewshot_eval` | `samples.jsonl` | `manifest.json` | 4,634,072 | 71,600 |

## 11. Split Count Validation
- Expected splits: 6.
- Actual splits generated: 6 (smoke, zero_shot_train, zero_shot_eval, fewshot_1pct_train, fewshot_5pct_train, fewshot_eval).
- **Verdict**: PASS.

## 12. Total Sample Count Validation
- Expected samples: 9,060.
- Actual samples generated: 9,060.
- **Verdict**: PASS.

## 13. Zero-shot C Holdout Validation
- `zero_shot_train` has `zero_shot_c_train_count == 0` in both runner result and manifest JSON.
- ARMA_GARCH counts in `zero_shot_train` family breakdown are 0.
- **Verdict**: PASS.

## 14. Fewshot 1%/5% Seed Superset Validation
- Fewshot 1% train C range: `[10_012_101, 10_012_110]`.
- Fewshot 5% train C range: `[10_012_101, 10_012_150]`.
- 1% seed range is verified to be a prefix subset of the 5% seed range.
- Summary Note: `"sample_id prefix not required because split/artifact identity differs; seed prefix semantics verified instead."`
- **Verdict**: PASS.

## 15. C Eval Holdout Non-Overlap Validation
- `zero_shot_eval` range: `[12003, 13002]`
- `fewshot_5pct_train` range: `[10_012_101, 10_012_150]`
- `fewshot_eval` range: `[20_012_105, 20_013_104]`
- Programmatic validations confirm ranges do not overlap.
- **Verdict**: PASS.

## 16. Manifest/Hash Validation
- All SHA256 hashes generated by P10 writer match the physical file hashes.
- Manifest files correctly specify `artifact_type = "phase2_dataset_manifest"` and include lists of sample IDs.
- **Verdict**: PASS.

## 17. Unexpected File Check
- No files exist outside of the `samples.jsonl` and `manifest.json` files within the 6 split directories under `phase2_artifacts/p14_smoke_dry_run/`.
- **Verdict**: PASS.

## 18. No Model/Training/Metrics/Config Confirmation
- No model architecture, neural network code, VAE, metric scoring, argparse CLI, or YAML loaders were added or modified.
- Standard library utilities are strictly maintained.

## 19. Artifact Commit Exclusion Confirmation
- The `phase2_artifacts/` directory is appended to `.gitignore`.
- No generated artifacts are staged or committed.

## 20. Remaining Blockers
None.

## 21. Post-Commit/Push Evidence
- Branch: `phase2/p14-smoke-artifact-generation-dry-run`
- Commit: `a61fdf591a5a51c925e3b599ab7f147008bd5f92`

## 22. Final Verdict
**P14_READY_FOR_REVIEW_NORMALIZED_BY_P15**
