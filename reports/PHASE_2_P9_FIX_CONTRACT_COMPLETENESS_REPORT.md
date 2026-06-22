# PHASE_2_P9_FIX_CONTRACT_COMPLETENESS_REPORT.md

## Branch and Base Commit

- **Fix Branch**: `phase2/p9-fix-contract-completeness`
- **Base Branch**: `phase2/p9-split-safe-in-memory-dataset-builder`
- **Base Commit**: `dab660bd8bc0ba5fb5f3dfeb261b474a30a4a2d4`
- **P8 head**: `4837e862e6a52e1c9220b67a863026acf188097e`

## Files Changed

| File | Action |
|------|--------|
| `src/phase2/dataset.py` | Modified — full contract fix |
| `tests/test_phase2_dataset.py` | Modified — full test matrix (all 40 items) |
| `reports/PHASE_2_P9_SPLIT_SAFE_IN_MEMORY_DATASET_BUILDER_REPORT.md` | New file — required P9 base report |
| `reports/PHASE_2_P9_FIX_CONTRACT_COMPLETENESS_REPORT.md` | New file — this report |

## Contract Fixes Applied

### Fix 1: DatasetSample expanded to full contract

**Before**: `DatasetSample` had only `sample_id`, `spec`, `values`, `innovations`, `variances`.

**After**: `DatasetSample` now contains all required fields:
```python
@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    protocol_name: str
    split_name: SplitName
    family_id: FamilyId
    sample_index: int
    generation_seed: int
    simulation_seed: int
    model_spec: ModelSpec          # renamed from spec
    values: tuple[float, ...]
    innovations: tuple[float, ...]
    variances: tuple[float, ...]
    provenance: tuple[tuple[str, str], ...]
```

### Fix 2: DatasetBuildResult expanded to full contract

**Before**: `DatasetBuildResult` had only `samples` and `split_name`.

**After**: `DatasetBuildResult` now contains all required fields:
```python
@dataclass(frozen=True)
class DatasetBuildResult:
    samples: tuple[DatasetSample, ...]
    protocol_name: str
    split_name: SplitName
    total_count: int
    count_by_family: tuple[tuple[FamilyId, int], ...]
    zero_shot_c_train_count: int
    reason: str
```

`reason` is always `"dataset_built_in_memory"`.
`zero_shot_c_train_count` is the actual count of `ARMA_GARCH` samples built.

### Fix 3: sample_id_hash_len minimum enforced at >= 8

**Before**: Any positive integer was accepted.

**After**: Validation rejects any `sample_id_hash_len < 8` with a clear error message. `make_sample_id` also enforces the 8-to-64 range.

### Fix 4: protocol_name non-empty enforced

**Before**: Empty string `""` or whitespace-only was accepted.

**After**: `validate_dataset_build_request` checks `len(request.protocol_name.strip()) == 0` and raises `ValueError`.

### Fix 5: validate_generation_request enforced per template

**Before**: Generation templates were only type-checked as `GenerationRequest` instances.

**After**: Each template is also passed through `validate_generation_request()`, which rejects invalid order configurations, bad range types, or family mismatches.

### Fix 6: validate_simulation_request enforced on simulation_template

**Before**: Simulation template was only type-checked.

**After**: The template is passed through `validate_simulation_request()`, which rejects invalid length, burn_in, distribution, std, variances, etc.

### Fix 7: Template family_id mismatch explicitly rejected

**Before**: No check on whether the GenerationRequest.family_id matched the tuple key.

**After**: Explicit check raises `ValueError` if `gen_req.family_id != f_id`.

### Fix 8: Zero total count explicitly rejected

**Before**: An all-zero count tuple would produce an empty result without error.

**After**: `validate_dataset_build_request` computes `total = sum(count for _, count in ...)` and raises `ValueError` if `total <= 0`.

### Fix 9: Augmented provenance in DatasetSample

**Before**: `DatasetSample.provenance` was not a field.

**After**: `DatasetSample.provenance` contains the original `ModelSpec.provenance` entries plus five builder-injected entries:
```python
("protocol_name", request.protocol_name)
("split_name", request.split_name.value)
("generation_seed", str(generation_seed))
("simulation_seed", str(simulation_seed))
("sample_index", str(sample_index))
```

The original `ModelSpec.provenance` is not mutated; tuple concatenation creates a new tuple.

### Fix 9: Removed unused Optional import

`from typing import Optional` was removed since it is not used in the updated code.

## Zero-Shot Leakage Evidence

`validate_dataset_build_request` check:
```python
if request.split_name == SplitName.ZERO_SHOT_TRAIN and request.enforce_zero_shot_c_train_exclusion:
    arma_garch_count = 0
    for f_id, count in request.sample_count_by_family:
        if f_id == FamilyId.ARMA_GARCH:
            arma_garch_count = count
    if arma_garch_count > 0:
        raise ValueError(
            f"C leakage detected: ARMA_GARCH count is {arma_garch_count} (> 0), "
            f"which is strictly prohibited in split zero_shot_train."
        )
```

- ARMA_GARCH omitted → `arma_garch_count` stays `0` → allowed
- ARMA_GARCH count == 0 → allowed
- ARMA_GARCH count > 0 → raises ValueError with "C leakage" and "zero_shot_train"
- Other splits not affected

## Sample ID Behavior

Format:
```
{family}_{split}_{index_zero_padded}_g{generation_seed}_s{simulation_seed}_{hash[:hash_len]}
```

Hash payload: canonical JSON (sort_keys=True, separators=(",", ":")) of:
`protocol_name`, `split_name`, `family_id`, `sample_index`, `generation_seed`, `simulation_seed`, `p`, `q`, `r`, `s`, `ar_params`, `ma_params`, `omega`, `alpha_params`, `beta_params`, `constraint_flags`, `provenance`

Enums serialized as `.value` strings. `hash_len` must be >= 8.

## Scope Confirmation

- No file output: `dataset.py` contains no `open(`, `.write(`, `.to_csv`, `.to_json`, `save(`, or `Path(` calls.
- No artifacts: No artifact writer instantiated or called.
- No model / training / metrics: `dataset.py` imports only from `src.phase2.schema`, `src.phase2.sampler`, `src.phase2.simulator`, `json`, `hashlib`, `dataclasses`.
- No config: No `configs` module referenced.
- No direct numpy: No `import numpy` or `from numpy` in `dataset.py`.
- No direct torch: Subprocess test confirms torch is not imported when loading `phase2.dataset`.

## Test Results

### Required Command

```
python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py -q
```

Result: **189 passed in 1.05s**

### Optional Compatibility Command

```
python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py -q
```

Result: **280 passed, 4 skipped in 1.22s**

## Post-Push Evidence

```
git log --oneline -3
git status --short
git ls-remote origin phase2/p9-fix-contract-completeness
```
(See below after push)

## Final Verdict

P9_FIX_READY_FOR_REVIEW
