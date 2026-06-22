# PHASE_2_P9_SPLIT_SAFE_IN_MEMORY_DATASET_BUILDER_REPORT.md

## Branch and Base Commit

- **Branch**: `phase2/p9-split-safe-in-memory-dataset-builder`
- **Commit**: `dab660bd8bc0ba5fb5f3dfeb261b474a30a4a2d4`
- **Base (P8 head)**: `4837e862e6a52e1c9220b67a863026acf188097e`
- **P8 code commit**: `fea7dcaacea78cc05a59699f69e5ccb8af08c837`

## Files Changed

| File | Action |
|------|--------|
| `src/phase2/dataset.py` | New file — split-safe in-memory dataset builder |
| `src/phase2/__init__.py` | Modified — added dataset exports |
| `tests/test_phase2_dataset.py` | New file — P9 test suite |

## P9 Implementation Summary

### SplitName

Five exact splits:
- `zero_shot_train`
- `zero_shot_eval`
- `fewshot_train`
- `fewshot_eval`
- `smoke`

### DatasetBuildRequest

Required fields (no defaults, no default_factory):
- `protocol_name: str`
- `split_name: SplitName`
- `sample_count_by_family: tuple[tuple[FamilyId, int], ...]`
- `generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...]`
- `simulation_template: SimulationRequest`
- `base_seed_by_family: tuple[tuple[FamilyId, int], ...]`
- `enforce_zero_shot_c_train_exclusion: bool`
- `sample_id_hash_len: int`

### Seed Policy

```
generation_seed = base_seed_by_family[family_id] + i  (i is 0-based)
simulation_seed = base_seed_by_family[family_id] + 1_000_000 + i
sample_index = i + 1  (1-based in output)
```

### Sample ID Format

```
{family}_{split}_{index_zero_padded}_g{generation_seed}_s{simulation_seed}_{hash}
```

Example:
```
ARMA_GARCH_zero_shot_eval_000001_g12003_s1012003_ab12cd34ef56
```

Hash payload (canonical JSON, sort_keys=True, separators=(",", ":")):
- `protocol_name`, `split_name`, `family_id`, `sample_index`, `generation_seed`, `simulation_seed`
- `p`, `q`, `r`, `s`
- `ar_params`, `ma_params`, `omega`, `alpha_params`, `beta_params`
- `constraint_flags`, `provenance`
- Enums serialized as `.value` strings

### Zero-Shot C Leakage Guard

If `split_name == ZERO_SHOT_TRAIN` and `enforce_zero_shot_c_train_exclusion == True`:
- ARMA_GARCH omitted → allowed (count defaults to 0)
- ARMA_GARCH count == 0 → allowed
- ARMA_GARCH count > 0 → raises `ValueError` containing "C leakage" and "zero_shot_train"

### No File Output

- `build_dataset_in_memory` returns only in-memory `DatasetBuildResult`
- No file writes, no JSONL/CSV/artifact output
- No `torch` or `numpy` direct imports in `dataset.py`
- No configs, no model, no training, no metrics

## Test Results

Required command:
```
python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py -q
```
Result: **150 passed in 1.61s**

## Post-Push Evidence

```
git log --oneline -3
dab660b PHASE2_P9 split-safe in-memory dataset builder
4837e86 Update report with post-push evidence
fea7dca PHASE2_P8 in-memory time-series simulator
```

```
git ls-remote origin phase2/p9-split-safe-in-memory-dataset-builder
dab660bd8bc0ba5fb5f3dfeb261b474a30a4a2d4  refs/heads/phase2/p9-split-safe-in-memory-dataset-builder
```
