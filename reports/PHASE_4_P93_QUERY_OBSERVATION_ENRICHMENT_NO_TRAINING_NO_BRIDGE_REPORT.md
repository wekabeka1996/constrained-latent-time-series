# PHASE 4 / P93 — Query Observation Enrichment Report

**NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P93_QUERY_OBSERVATION_ENRICHMENT_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p93-query-observation-enrichment-no-training-no-bridge`

Based on accepted P92 head: `85ae77db2c891f20a415d52a69dbc0f3a0f016ec`

---

## 1. Problem Framing

P92 established that a simple source-similarity distance helper improves non-label-selected retrieval to 38.89% accuracy, but still falls short of the 40% target signal threshold because the query's raw source features are underdetermined. P93 asks: can richer deterministic source-side query observation enrichment help non-label-selected retrieval approximate same-relation support selection?

## 2. P92 Inheritance

| P92 Baseline / Improved Result | Value |
|---|---|
| p91_observable_domain_split_retrieval_baseline | 22.22% |
| p91_external_manifest_support_baseline | 22.22% |
| source_similarity_retrieval | 38.89% |
| source_similarity_with_split_relaxation | 38.89% |
| RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT | False |
| IMPROVED_NON_LABEL_SELECTED_SUPPORT_READY | True |
| Bridge ready | False |

P93 validates and preserves P92's near-threshold source-similarity accuracy (38.89%) and the absence of a retrieval signal in P92.

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |

P93 does not train a model. P93 does not prove learned metric evidence. P93 does not prove semantic geometry. P93 does not validate any bridge. P93 enriches observable query/source context only.

## 4. Enrichment Principle

Query observations are enriched strictly using source-side features. All target relation labels, operator IDs, query targets, query deltas, result summaries of the query, and support result/delta content are forbidden for retrieval scoring.

## 5. Enriched Query Schema

Enriched query records are built with the following schema:
- `enriched_query_id`
- `external_query_id`
- `dataset_record_id`
- `domain`
- `split`
- `query_observation`
- `enriched_query_observation`
  - `source_geometry_fingerprint`
  - `source_sparsity_signature`
  - `source_magnitude_signature`
  - `source_sign_or_direction_signature`
  - `domain_normalized_source_signature`
  - `observable_context_bucket`
- `enrichment_audit` (ensuring no label/operator/target/delta/result use)
- `audit_label_evaluation_only`

## 6. Enriched Support Metadata Schema

Enriched support metadata records are built with:
- `external_demo_id`
- `domain`
- `source_split`
- `support_source_observation`
- `enriched_support_metadata`
- `selection_visible_metadata`
- `retrieval_scoring_audit` (ensuring no label/operator/result/delta content/delta shape usage)
- `audit_label_evaluation_only`

## 7. Enriched Retrieval Policies

### Baselines (from P92)

1. `p92_source_similarity_retrieval_baseline` (38.89% in P92, but evaluated here on enriched records)
2. `p92_source_similarity_with_split_relaxation_baseline`

### Improved (P93 Enriched)

3. `enriched_source_signature_retrieval`: Score support using enriched source distance.
4. `enriched_source_signature_with_split_relaxation`: Cross-split fallback for relaxed signature matching.
5. `context_bucket_retrieval`: Retrieve support from the same context bucket.
6. `hybrid_enriched_signature_bucket_retrieval`: Combine signature distance and context bucket alignment.
7. `cross_domain_normalized_signature_retrieval` (Diagnostic only): Match support across domains using normalized features.

## 8. Metric Results

Evaluated using the same P91/P92 metric (18-dimensional support-delta vector, L2 nearest prototype):

| Policy | Effective Accuracy | Coverage | Supported Accuracy | Test Ctx | Supported | Unsupported | Diagnostic |
|---|---|---|---|---|---|---|---|
| p92_source_similarity_retrieval_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ |
| p92_source_similarity_with_split_relaxation_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ |
| **enriched_source_signature_retrieval** | **38.89%** | 100.00% | 38.89% | 18 | 18 | 0 | ❌ |
| **enriched_source_signature_with_split_relaxation** | **38.89%** | 100.00% | 38.89% | 18 | 18 | 0 | ❌ |
| context_bucket_retrieval | 33.33% | 100.00% | 33.33% | 18 | 18 | 0 | ❌ |
| **hybrid_enriched_signature_bucket_retrieval** | **38.89%** | 100.00% | 38.89% | 18 | 18 | 0 | ❌ |
| cross_domain_normalized_signature_retrieval | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 | ✅ (diag only) |

The best enriched policies match the P92 maximum of 38.89% effective accuracy, but do not exceed it.

## 9. Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| p92_source_similarity_retrieval_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| p92_source_similarity_with_split_relaxation_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| enriched_source_signature_retrieval | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| enriched_source_signature_with_split_relaxation | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| context_bucket_retrieval | 33.33% | 0.00% | 11.11% | +33.33% | +22.22% |
| hybrid_enriched_signature_bucket_retrieval | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| cross_domain_normalized_signature_retrieval | 22.22% | 5.56% | 11.11% | +16.67% | +11.11% |

### Control Properties

- shuffled_control_target_labels_preserved = true
- shuffled_control_support_deltas_permuted = true
- target_labels_shifted = false
- true_labels_preserved = true

## 10. Leakage Audits

All policies pass the enrichment leakage audit:
- Query enrichment uses label/operator/target/delta/result = 0
- Support metadata uses label/operator/result/delta = 0
- Selection uses label/operator/target/delta/result = 0
- `diagnostic_pass` = True for all policies.

## 11. Proxy Leakage Audit

Proxy audit flags relation purity over enriched fields:
- Enriched query context bucket and sign bucket purities are clean (all < 95% threshold).
- Enriched support metadata is clean.
- Policy selection distribution is flagged for proxy leakage (risk present), ensuring semantic geometry claims remain false.
- `proxy_leakage_risk_present` = True (due to policy selection distribution).

## 12. Enrichment Verdict

### Signal Criteria

For ENRICHED_RETRIEVAL_SIGNAL_PRESENT = True:
- effective_accuracy ≥ 40%
- coverage ≥ 80%
- beats shuffled ≥ 10%
- beats zero-delta ≥ 10%
- leakage clean
- shape compatible

Since the best enriched policy achieves **38.89%** effective accuracy (below 40%):

- `QUERY_OBSERVATION_ENRICHMENT_READY` = **True** (all record builders run, audits clean)
- `ENRICHED_RETRIEVAL_SIGNAL_PRESENT` = **False**
- `ENRICHED_EXTERNAL_CONTEXT_METRIC_SUPPORTED` = **False**

## 13. Recommended Next Phase

`P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge`

Enriching query observations with source-side features is insufficient to cross the 40% threshold. The relation mapping is structurally underdetermined from the source alone. P94 must enrich the query observation with true external task demonstrations or task-level context (e.g. prompt-side task definitions or demonstration-visible context) rather than source statistics.

## 14. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. learned_metric_evidence_not_present
2. semantic_metric_not_ready
3. external_context_metric_not_supported_for_bridge
4. bridge_input_contract_not_defined
5. bridge_validation_not_run

## 15. Limitations

- Source-side features cannot resolve relation identity with high accuracy because multiple relations can share identical vector sign patterns or time-series parameter statistics.
- Relaxation across splits doesn't boost signature accuracy beyond 38.89%.
- The small sample size (18 test queries) results in coarse metric percentages.

## 16. Final Verdict

**P93_READY_FOR_REVIEW**
