# PHASE 2 ARTIFACT CONTRACT
# Required Artifact Structure and Manifest Requirements

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval

> [!IMPORTANT]
> This document specifies the required artifact structure for future Phase 2 implementation.
> **The directory described here does not exist and must not be created now.**
> This is a specification only. The artifact directory will be created by the Phase 2 implementation
> task (Phase 2 P3) and populated by Phase 2 training/evaluation runs.

---

## 1. Required Artifact Directory Structure

Each Phase 2 experiment run must produce the following artifact structure:

```
artifacts/phase2/runs/<run_id>/
├── manifest.json                  REQUIRED — run-level manifest (see §3)
├── config_snapshot.yaml           REQUIRED — exact copy of config used for this run
├── git_commit.txt                 REQUIRED — full SHA of HEAD at run time
├── dirty_tree_status.txt          REQUIRED — clean/dirty plus git status output
├── environment_freeze.txt         REQUIRED — pip freeze output, python/torch version, etc.
├── dataset_manifest.json          REQUIRED — all sample IDs and family labels used
├── split_manifest.json            REQUIRED — split assignment for every sample
├── metric_contract_snapshot.md    REQUIRED — copy of docs/PHASE_2_METRIC_CONTRACT.md at commit time
├── metric_contract_hash.txt       REQUIRED — SHA-256 hash of docs/PHASE_2_METRIC_CONTRACT.md
├── split_contract_hash.txt        REQUIRED — SHA-256 hash of docs/PHASE_2_SPLIT_CONTRACT.md
├── schema_spec_hash.txt           REQUIRED — SHA-256 hash of docs/PHASE_2_SCHEMA_SPEC.md
├── protocol_hash.txt              REQUIRED — SHA-256 hash of docs/PHASE_2_PROTOCOL.md
├── model_card.md                  REQUIRED — model architecture, training config, training seed
├── metrics.json                   REQUIRED — all metric values computed (see §4)
├── verdict.json                   REQUIRED — pass/fail verdict against precommitted thresholds
├── report.md                      REQUIRED — human-readable run summary
├── generated_samples.parquet      REQUIRED — generated C-candidate samples (or .csv if parquet not available)
└── failure_cases.csv              REQUIRED — samples that failed validity checks or other assertions
```

**Notes on run_id:**
- `run_id` must be a globally unique identifier for each run
- APPROVED_PROTOCOL_VALUE: run_id format = `phase2_<stage>_<protocol>_<YYYYMMDD_HHMMSS>_<gitshort>_<seedgroup>` (e.g., `phase2_p5_generator_smoke_20260620_193000_1e3bb97_gen12001`)
- Must be deterministically reproducible from: seed list + config hash + git commit hash + timestamp
- The run_id must be the same across all artifact files in the same run

**Notes on path roots:**
- `artifacts/` is a repository-relative path
- `artifacts/phase2/runs/<run_id>/` must not be committed to git if it contains large binary files
- The `.gitignore` must be updated to exclude large artifact files but NOT the manifest files
- Specifically: `generated_samples.parquet`, `environment_freeze.txt` may be gitignored
  but `manifest.json`, `git_commit.txt`, `metrics.json`, `verdict.json` should be committed

---

## 2. Do Not Create This Directory Now

This artifact directory structure must not be created as part of this P2 documentation task.
Creating the directory now would be premature and would violate the documentation-only constraint.

The following actions are explicitly prohibited in this task:
- Creating `artifacts/`
- Creating `artifacts/phase2/`
- Creating any `run_id/` directory
- Creating any `manifest.json`
- Creating any checkpoint
- Creating any dataset file

---

## 3. Manifest Requirements

`manifest.json` must contain the following fields. If any field is missing, the run is invalid.

```json
{
  "run_id": "<unique run identifier>",
  "git_commit": "<full 40-character SHA of HEAD at run time>",
  "git_branch": "<branch name at run time>",
  "git_dirty": <true|false>,         // true if any tracked file was modified at run time
  "config_hash": "<SHA-256 of config_snapshot.yaml>",
  "metric_contract_hash": "<SHA-256 of metric_contract_snapshot.md>",
  "dataset_manifest_hash": "<SHA-256 of dataset_manifest.json>",
  "split_manifest_hash": "<SHA-256 of split_manifest.json>",
  "seed_list": [<seed_a>, <seed_b>, <seed_c>, <training_seed>],
  "model_checkpoint_hash": "<SHA-256 of model checkpoint file, or null if no checkpoint saved>",
  "environment_freeze_hash": "<SHA-256 of environment_freeze.txt>",
  "command_line": "<exact command line invocation as a string>",
  "start_timestamp": "<ISO 8601 UTC>",
  "end_timestamp": "<ISO 8601 UTC>",
  "python_version": "<e.g. 3.14.3>",
  "torch_version": "<e.g. 2.x.y>",
  "protocol_version": "phase2_p2_v1",
  "experiment_type": "<one of: zero_shot_c, few_shot_1pct, few_shot_5pct, supervised_upper_bound>",
  "split_contract_version": "phase2_p2_v1",
  "metric_contract_version": "phase2_p2_v1",
  "C_count_in_zero_shot_train": <int>,    // MUST be 0 for zero-shot experiments
  "artifact_completeness_score": <float>, // 1.0 means all required artifacts present
  "run_outcome": "<one of: PASS, FAIL_VALIDITY, FAIL_NOVELTY, FAIL_COMPOSITION, FAIL_SPLIT_VIOLATION, FAIL_ARTIFACT_INCOMPLETE, FAIL_OTHER>",
  "notes": "<optional string>"
}
```

### Manifest Validation Rules

Before a run is considered complete:
1. `git_dirty` must be `false` — no uncommitted changes at run time
2. `C_count_in_zero_shot_train` must be `0` for zero-shot experiments
3. `artifact_completeness_score` must be `1.0`
4. `metric_contract_hash` must match the hash of the approved metric contract commit
5. `split_manifest_hash` must match the hash of the pre-committed split manifest
6. All seed values must be present and non-null

---

## 4. Metrics File Requirements

`metrics.json` must contain all metrics defined in `docs/PHASE_2_METRIC_CONTRACT.md` §1–§4.
If any metric cannot be computed, the field must be present with value `null` and the
`failure_cases.csv` must document why.

```json
{
  "input_validity": {
    "A_input_valid_rate": <float or null>,
    "B_input_valid_rate": <float or null>,
    "C_input_valid_rate": <float or null>
  },
  "generated_C": {
    "generated_C_valid_rate": <float or null>,
    "generated_C_composition_score": <float or null>,
    "generated_C_novelty_score": <float or null>,
    "generated_C_distance_to_A": <float or null>,
    "generated_C_distance_to_B": <float or null>,
    "generated_C_distance_to_true_C_distribution": <float or null>
  },
  "series_diagnostics": {
    "simulation_failure_rate": <float or null>,
    "acf_pacf_mean_dynamics_pass_rate": <float or null>,
    "volatility_clustering_proxy_mean": <float or null>,
    "conditional_variance_max_mean": <float or null>
  },
  "reproducibility": {
    "seed_list": [<seeds used>],
    "seed_stability_range": <float or null>,
    "artifact_completeness_score": <float>
  }
}
```

---

## 5. Verdict File Requirements

`verdict.json` must contain the pass/fail verdict against each precommitted threshold.

```json
{
  "verdict": "<PASS | FAIL>",
  "thresholds_used": {
    "generated_C_valid_rate_threshold": <float>,
    "generated_C_composition_score_threshold": <float>,
    "generated_C_novelty_score_threshold": <float>,
    "seed_stability_range_threshold": <float>
  },
  "threshold_source": "<hash of metric_contract_snapshot.md>",
  "per_metric_verdict": {
    "generated_C_valid_rate": "<PASS | FAIL | MISSING>",
    "generated_C_composition_score": "<PASS | FAIL | MISSING>",
    "generated_C_novelty_score": "<PASS | FAIL | MISSING>",
    "seed_stability_range": "<PASS | FAIL | MISSING>",
    "split_contract_verified": "<PASS | FAIL>",
    "artifact_completeness": "<PASS | FAIL>"
  },
  "h0_outcome": "<CONFIRMED | FALSIFIED | INCONCLUSIVE>",
  "h1_outcome": "<SUPPORTED | NOT_SUPPORTED | INCONCLUSIVE>",
  "h2_outcome": "<SUPPORTED | NOT_SUPPORTED | INCONCLUSIVE | NOT_TESTED>",
  "negative_result": <true|false>,
  "claim_authorized": <true|false>
}
```

**`claim_authorized` is `true` only if ALL per_metric_verdicts are PASS.**
A partially passing run may not make a positive C-generation claim.

---

## 6. Required Artifact Contents Detail

### config_snapshot.yaml
- Exact copy of the config file used for the run
- Must be written before any computation begins and must not be modified during the run
- Must match `config_hash` in `manifest.json`

### git_commit.txt
- Single line: full 40-character commit SHA (no newlines or extra text)
- Must be written before any computation begins
- Must be reproducible via `git rev-parse HEAD` at run time
- This resolves P1 Blocker B2: `code_git_commit: None` hardcoded in manifest

### dirty_tree_status.txt
- First line: "clean" or "dirty" indicating if there are uncommitted changes in tracked files
- Remaining lines: output of `git status --short` at run time

### environment_freeze.txt
- Output of `pip freeze` at run time
- First line comment: `# Python <version>, platform <platform>, CUDA <available_boolean>, torch <torch_version_if_installed>`
- This resolves P1 Blocker B3: missing lock file

### metric_contract_hash.txt
- Single line: SHA-256 hash of the local `docs/PHASE_2_METRIC_CONTRACT.md` file at run time

### split_contract_hash.txt
- Single line: SHA-256 hash of the local `docs/PHASE_2_SPLIT_CONTRACT.md` file at run time

### schema_spec_hash.txt
- Single line: SHA-256 hash of the local `docs/PHASE_2_SCHEMA_SPEC.md` file at run time

### protocol_hash.txt
- Single line: SHA-256 hash of the local `docs/PHASE_2_PROTOCOL.md` file at run time

### dataset_manifest.json
- One entry per training and evaluation sample
- Each entry: `{"sample_id": <str>, "family_id": <str>, "split": <str>, "seed": <int>, "generator_mode": <str>}`
- Must include all A-train, B-train, C-holdout samples
- Must not include any samples with `family_id = "UNKNOWN"`

### split_manifest.json
- Documents the exact split assignment
- Must include: `C_count_in_zero_shot_train`, `A_train_count`, `B_train_count`, `C_holdout_count`
- Must include the precommitted split contract version
- Must be pre-committed and hashed before training begins

### generated_samples.parquet
- One row per generated C-candidate sample
- Columns: `sample_id`, `family_id` (proposed), `is_valid_C`, `ar_stationary`, `ma_invertible`, `garch_positive`, `garch_persistence_ok`, `composition_score`, `novelty_score`, and all parameter values
- If parquet is not available: use CSV

### failure_cases.csv
- One row per failed sample (validity failure, split violation, metric computation error)
- Columns: `sample_id`, `failure_type`, `failure_detail`, `step` (generation/training/evaluation)
- If no failures: an empty CSV with column headers must still be written (to confirm the file exists)

---

## 7. Artifact Discipline Rules

1. **No artifact overwriting.** Once written, an artifact must not be overwritten. Any re-run must produce a new `run_id` and a new artifact directory.
2. **No partial artifacts.** A run that does not produce all required artifacts is a FAIL.
3. **No cached artifacts in Phase 2.** `integrity.allow_cached_artifacts: false` must be enforced.
4. **Artifact provenance.** Every artifact must have its run_id embedded or traceable via manifest.
5. **No post-hoc artifact modification.** After a run completes, no artifact may be modified or deleted, including failure artifacts.
6. **Failure artifacts are valid artifacts.** A run that fails validation is still required to produce complete artifacts documenting the failure.
