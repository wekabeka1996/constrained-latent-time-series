# Phase 2 Model-to-Evidence Contract

This document defines how future neural model candidate outputs will map into P23-compatible normalized evidence bundles to ensure safe, objective, and consistent baseline comparisons.

---

## 1. Model Candidate Record

Every generated model candidate instance must produce an internal tracking record containing the following conceptual fields:
- `candidate_id`: A unique string identifier for the generated sample.
- `model_architecture_id`: Canonical name of the candidate model architecture (e.g. `FC-VAE`).
- `model_repeat_seed`: The specific initialization and sampling seed used.
- `generated_family_id`: The generated family classifier (e.g., `ARMA_GARCH`).
- `generated_mean_family`: The generated mean-dynamics type classification.
- `generated_volatility_family`: The generated volatility-dynamics type classification.
- `validity_pass`: Boolean indicating if the candidate passed all `validate_model_spec` and `require_math_valid` rules.
- `composition_score`: Quantitative composition score (0 or 1 for pass/fail, or rate).
- `novelty_score`: Quantitative novelty score relative to training references.
- `source_split`: The identifier of the training/validation data split used.
- `zero_shot_mode`: Boolean flag indicating if training strictly excluded composite C samples.

---

## 2. Evidence Mapping

To maintain compatibility without modifying the accepted P23 `evidence.py` in P24, model results must be converted conceptually into an equivalent baseline representation:
- The model outputs will be aggregated into a structure equivalent to `EvidenceBaselineRecord`.
- The generation properties (number of candidates, family list, seeds, and source indices) must map to `EvidenceGenerationRecord`.
- The evaluated metrics (validity counts/rates, composition counts/rates, novelty counts/rates, and distribution distances) must map directly to `EvidenceMetricRecord`.
- The final output is serialized as a baseline entry under the name of the model architecture inside `EvidenceBundle.baselines`.

---

## 3. Comparison Rules

To ensure scientifically valid comparisons between baselines and neural model candidates, the following rules are enforced:
- **Identical Environment**: Baseline and model candidates must be evaluated using the exact same `artifact_root`, train/test splits, and data orderings (TemporalLock-equivalent).
- **Identical Reference Extraction**: Reference specs must be extracted using the same rules and sample sizes.
- **Identical Metric Primitives**: All evaluations must use the identical metrics implemented in `src/phase2/metrics.py`.
- **Identical Contract Version**: Comparisons must use the same `evidence_contract_version` (e.g. `"phase2_p23_evidence_contract_v1"`). Any contract update requires a formal migration.
- **Objectivity**: No free-text "winner" or "best model" claims are permitted. All comparisons must consist of objective, side-by-side metric tables in the normalized evidence bundle.

---

## 4. Required Safety Checks

All generated model evidence records must pass the strict safety filters defined in P23:
- **No Raw Parameters**: Ensure **no raw params** (such as parameter coefficients or keys like `ar_params`, `omega`, etc.) leak into the final evidence record in key-value structure.
- **No Local Paths**: The record must contain no absolute paths (such as file scheme, drive letters, /home, or /Users prefixes).
- **No Forbidden Claims**: Enforce **no forbidden claims** containing marketing or non-objective success labels (e.g., subjective achievements, "best", "winner").
- **No Family C Train Leakage**: Ensure **no C leakage** (C_train_count must be exactly 0 in zero-shot mode).
- **No Hidden Defaults**: All configuration parameters must be explicitly set and hashed.

---

## 5. Future Extension Proposal

A future version of the evidence contract is proposed to explicitly distinguish baselines from neural models.

### Proposed `EvidenceModelRecord`
```python
# Conceptual extension only - do not implement in P24
@dataclasses.dataclass(frozen=True)
class EvidenceModelRecord:
    model_architecture_id: str
    run_metadata: dict  # Contains commit hash, config hash, seed
    generation: EvidenceGenerationRecord
    metrics: EvidenceMetricRecord
    zero_shot_mode: bool
    C_train_count: int
    reason: str
```

### Proposed `EvidenceComparisonBundle`
```python
# Conceptual extension only - do not implement in P24
@dataclasses.dataclass(frozen=True)
class EvidenceComparisonBundle:
    contract_version: str
    metadata: EvidenceRunMetadata
    baselines: Tuple[EvidenceBaselineRecord, ...]
    models: Tuple[EvidenceModelRecord, ...]
    verdict: str
```
This future structure will prevent baselines and models from being mixed in the same arrays, enabling cleaner multi-candidate comparison reporting.
