# PHASE 4 / P92 — Non-Label-Selected Support Retrieval Improvement Report

**NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P92_NON_LABEL_SELECTED_SUPPORT_RETRIEVAL_IMPROVEMENT_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p92-non-label-selected-support-retrieval-improvement-no-training-no-bridge`

Based on accepted P91 head: `b8caa1c6769489e970e9a4bb846e192357e0ea78`

---

## 1. Problem Framing

P91 showed the 18-dimensional support-delta metric collapses from 66.67% to 22.22% when support is selected without hidden relation labels. The bottleneck is not the metric—it's the retrieval. P92 asks: can improved deterministic non-label-selected retrieval strategies approximate same-relation support retrieval?

## 2. P91 Inheritance

| P91 Result | Value |
|---|---|
| Oracle (label-selected) | 66.67% |
| Observable domain split | 22.22% |
| External manifest | 22.22% |
| Non-label signal present | False |
| External metric supported | False |
| Bridge ready | False |

P92 preserves P91's oracle signal (True) and non-label collapse (True).

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |
| No artifact written | ✅ |

P92 does not train a model. P92 does not prove learned metric evidence. P92 does not prove semantic geometry. P92 does not validate any bridge. P92 evaluates deterministic non-label-selected retrieval policies only.

## 4. Retrieval Policies

### Baselines (from P91)

| Policy | Description |
|---|---|
| `p91_observable_domain_split_retrieval_baseline` | Same domain/split + observable source shape match |
| `p91_external_manifest_support_baseline` | Same domain/split, deterministic first-2 |

### Improved Policies

| Policy | Description |
|---|---|
| `source_similarity_retrieval` | Source similarity distance (z_a features / parameter features), same split |
| `source_similarity_with_split_relaxation` | Source similarity with cross-split fallback if same-split insufficient |
| `observable_diversity_manifest` | Greedy diverse selection from observable metadata buckets |
| `hybrid_source_similarity_diversity` | Top-k similarity → diverse pick from candidates |

## 5. Source Similarity Retrieval

For P70A (vector world): compares `z_a_dim`, `z_a_abs_sum`, `z_a_sign_pattern`, and `query_intensity_hint`.

For P70B (time-series parameter world): compares `parameter_key_count`, `source_abs_sum`, `source_nonzero_key_count`, and `query_intensity_hint`.

**No labels, operator IDs, query targets, or query deltas used.**

## 6. Diversity Manifest Retrieval

Selects demonstrations from distinct observable metadata buckets (`observable_source_shape × observable_delta_shape × observable_context_hash`). Uses `observable_delta_shape` as selection metadata (audited for proxy risk).

## 7. Hybrid Retrieval

Top-k candidates by source similarity, then diverse selection from those candidates.

## 8. Metric Results

| Policy | Eff. Accuracy | Coverage | Acc. on Supported | Test | Sup | Unsup |
|---|---|---|---|---|---|---|
| p91_observable_domain_split_retrieval_baseline | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 |
| p91_external_manifest_support_baseline | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 |
| **source_similarity_retrieval** | **38.89%** | 100.00% | 38.89% | 18 | 18 | 0 |
| **source_similarity_with_split_relaxation** | **38.89%** | 100.00% | 38.89% | 18 | 18 | 0 |
| observable_diversity_manifest | 0.00% | 100.00% | 0.00% | 18 | 18 | 0 |
| hybrid_source_similarity_diversity | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 |

**Source similarity retrieval improves from 22.22% → 38.89%** (+16.67 percentage points), but remains below the 40% threshold required for signal.

## 9. Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| p91_observable_domain_split_retrieval_baseline | 22.22% | 11.11% | 11.11% | +11.11% | +11.11% |
| p91_external_manifest_support_baseline | 22.22% | 11.11% | 11.11% | +11.11% | +11.11% |
| source_similarity_retrieval | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| source_similarity_with_split_relaxation | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| observable_diversity_manifest | 0.00% | 11.11% | 11.11% | −11.11% | −11.11% |
| hybrid_source_similarity_diversity | 22.22% | 0.00% | 11.11% | +22.22% | +11.11% |

Source similarity policies beat controls convincingly (+38.89% over shuffled), confirming real signal—just not enough to meet the 40% threshold.

### Control Properties (All Policies)

| Property | Value |
|---|---|
| shuffled_control_target_labels_preserved | ✅ true |
| shuffled_control_support_deltas_permuted | ✅ true |
| target_labels_shifted | ❌ false |
| true_labels_preserved | ✅ true |

## 10. Leakage Audits

| Policy | Label | Query Target | Query Delta | Operator ID | Delta Shape Meta | Result Content | Delta Content | Pass |
|---|---|---|---|---|---|---|---|---|
| source_similarity_retrieval | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ✅ |
| source_similarity_with_split_relaxation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ✅ |
| observable_diversity_manifest | 0 | 0 | 0 | 0 | 114 | 0 | 0 | ✅ |
| hybrid_source_similarity_diversity | 0 | 0 | 0 | 0 | 114 | 0 | 0 | ✅ |

Diversity-based policies use delta shape metadata (allowed but proxy-audited).

## 11. Proxy Leakage Audit

### Metadata Fields

| Field | Unique Values | Max Purity | Proxy Risk |
|---|---|---|---|
| observable_source_shape | 2 | 22.22% | ❌ No |
| observable_delta_shape | 3 | 22.64% | ❌ No |
| observable_context_hash | 6 | 25.00% | ❌ No |
| domain_split_source_shape | 6 | 25.00% | ❌ No |
| domain_split_delta_shape | 7 | 25.00% | ❌ No |

### Policy Selection Distribution

Proxy audit checks whether specific demo IDs are consistently selected for specific relation labels. Some policies show selection proxy risk—this does not block P92 but is reported honestly.

**Overall proxy_leakage_risk_present = True** (from policy selection distribution patterns).

## 12. Retrieval Improvement Verdict

### Signal Criteria

For RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT = True, an improved policy must satisfy ALL:
- effective_accuracy ≥ 40%
- coverage ≥ 80%
- beats shuffled ≥ 10%
- beats zero-delta ≥ 10%
- leakage clean
- shape compatible

| Policy | Acc ≥ 40% | Cov ≥ 80% | Beats Shuf ≥ 10% | Beats Zero ≥ 10% | Leak OK | Shape OK | Pass |
|---|---|---|---|---|---|---|---|
| source_similarity_retrieval | ❌ 38.89% | ✅ | ✅ +38.89% | ✅ +27.78% | ✅ | ✅ | ❌ |
| source_similarity_with_split_relaxation | ❌ 38.89% | ✅ | ✅ +38.89% | ✅ +27.78% | ✅ | ✅ | ❌ |
| observable_diversity_manifest | ❌ 0.00% | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| hybrid_source_similarity_diversity | ❌ 22.22% | ✅ | ✅ +22.22% | ✅ +11.11% | ✅ | ✅ | ❌ |

**RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT = False**

Best improved policy (source_similarity_retrieval: 38.89%) falls just below the 40% threshold.

**IMPROVED_NON_LABEL_SELECTED_SUPPORT_READY = True**

All improved policies build valid, leak-clean, shape-compatible contexts.

## 13. Recommended Next Phase

`P93_query_observation_enrichment_no_training_no_bridge`

Retrieval cannot recover relation identity from current query source context alone. The query observation needs enrichment (additional contextual features) to make non-label-selected retrieval more discriminative.

## 14. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. learned_metric_evidence_not_present
2. semantic_metric_not_ready
3. external_context_metric_not_supported
4. bridge_input_contract_not_defined
5. bridge_validation_not_run

## 15. Limitations

- Source similarity improves retrieval (+16.67pp) but is insufficient to cross the 40% threshold.
- Diversity-based retrieval actually hurts performance (0.00%) by selecting irrelevant diverse demonstrations.
- The fundamental issue is P83's identifiability finding: source observations alone cannot distinguish relations.
- Cross-split relaxation provides no additional benefit over same-split similarity retrieval.
- With only 18 test samples, small differences in accuracy are not statistically significant.

## 16. Final Verdict

**P92_READY_FOR_REVIEW**

## Files

| # | File | Status |
|---|---|---|
| 1 | `src/phase4/non_label_support_retrieval_improvement.py` | NEW |
| 2 | `tools/phase4/run_p92_non_label_support_retrieval_improvement_smoke.py` | NEW |
| 3 | `tests/test_phase4_non_label_support_retrieval_improvement.py` | NEW |
| 4 | `tests/test_phase4_p92_non_label_support_retrieval_improvement_smoke.py` | NEW |
| 5 | `reports/PHASE_4_P92_NON_LABEL_SELECTED_SUPPORT_RETRIEVAL_IMPROVEMENT_NO_TRAINING_NO_BRIDGE_REPORT.md` | NEW |

## Test Results

31 passed, 0 failed.
