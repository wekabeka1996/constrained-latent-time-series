# PHASE 4 / P94 — External Task Context or Support Demonstration Enrichment Report

**NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P94_EXTERNAL_TASK_CONTEXT_OR_SUPPORT_DEMONSTRATION_ENRICHMENT_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p94-external-task-context-or-support-demonstration-enrichment-no-training-no-bridge`

Based on accepted repaired P93 head: `c05b7b108d5de7d7e6abfb006405c5f0b5ae205e`

---

## 1. Problem Framing

P93 established that source-side query enrichment cannot break the source-similarity accuracy ceiling of 38.89% because source observations are underdetermined. P94 investigates the next potential channels: external task context or support demonstration context. We evaluate whether a non-label task context channel can be represented, audited, and evaluated without leaking relation identity.

## 2. P93 Repaired Inheritance

P94 validates and inherits the repaired P93 baseline and improved values:
- `p92_source_similarity_retrieval_baseline` = 38.89%
- `p92_source_similarity_with_split_relaxation_baseline` = 38.89%
- `enriched_source_signature_retrieval` = 38.89%
- `enriched_source_signature_with_split_relaxation` = 38.89%
- `best_enriched_effective_accuracy` = 38.89%
- `query_observation_enrichment_ready` = True
- `enriched_retrieval_signal_present` = False
- `enriched_external_context_metric_supported` = False

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |

P94 does not train a model. P94 does not prove learned metric evidence. P94 does not prove semantic geometry. P94 does not validate any bridge. P94 separates diagnostic oracle task context from valid non-label external task context.

## 4. Context Type Taxonomy

P94 separates four context channels:

| Context Type | Description | Uses Label | Valid for Evidence |
|---|---|---|---|
| `task_context_absent` | Baseline without external context (equivalent to P93). | ❌ No | ✅ Yes |
| `diagnostic_oracle_task_context` | Identity context using target relation label/operator ID. | ✅ Yes | ❌ No (Diag only) |
| `non_label_external_task_manifest` | Public descriptor card ID/hash/bucket independent of query labels. | ❌ No | ✅ Yes |
| `support_demonstration_visible_context` | Support content descriptors (source, result, delta, invariants). | ❌ No | ✅ Yes |

## 5. External Task Context Schema

Task context records contain:
- `task_context` (available, context_source, public_task_card_id, public_task_bucket, public_task_descriptor)
- `task_context_audit` (uses_hidden_relation_label = False, valid_for_non_label_retrieval = True)
- `diagnostic_task_context` (for oracle task retrieval)
- `diagnostic_task_context_audit` (uses_hidden_relation_label = True, diagnostic_only = True)

## 6. Support Demonstration Context Schema

Demonstration context records contain:
- `support_demonstration_context` (support_source_descriptor, support_result_descriptor, support_delta_descriptor, support_invariant_descriptor, support_demo_descriptor_hash)
- `support_demo_context_audit` (uses_hidden_relation_label = False, valid_as_demonstration_context = True)

## 7. Policies Evaluated

### Baselines
1. `p93_enriched_source_signature_baseline`: P93 best signature retrieval.
2. `task_context_absent_baseline`: Baseline with no external task context.

### Diagnostic Oracle Policies
3. `diagnostic_oracle_task_context_retrieval`: Selects support demonstrations matching hidden relation label.
4. `diagnostic_support_delta_oracle_retrieval`: Selects support demonstrations matching target delta.

### Non-Label Candidate Policies
5. `non_label_external_task_manifest_retrieval`: Selects using public card manifest ordering.
6. `task_manifest_plus_source_similarity_retrieval`: Combines manifest alignment and signature matching.
7. `support_demo_context_without_query_task_card`: Uses support demo context without query card.
8. `support_demo_context_with_non_label_task_manifest`: Combines support descriptors and manifest alignment.

## 8. Metric Results

| Policy | Effective Accuracy | Coverage | Supported Accuracy | Test Ctx | Supported | Unsupported | Diagnostic | Valid |
|---|---|---|---|---|---|---|---|---|
| p93_enriched_source_signature_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ | ✅ |
| task_context_absent_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ | ✅ |
| **diagnostic_oracle_task_context_retrieval** | **66.67%** | 100.00% | 66.67% | 18 | 18 | 0 | ✅ | ❌ |
| **diagnostic_support_delta_oracle_retrieval** | **66.67%** | 100.00% | 66.67% | 18 | 18 | 0 | ✅ | ❌ |
| non_label_external_task_manifest_retrieval | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 | ❌ | ✅ |
| task_manifest_plus_source_similarity_retrieval | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ | ✅ |
| support_demo_context_without_query_task_card | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 | ✅ | ❌ |
| support_demo_context_with_non_label_task_manifest | 22.22% | 100.00% | 22.22% | 18 | 18 | 0 | ✅ | ❌ |

Note: Support demonstration descriptors are built and audited, so `SUPPORT_DEMONSTRATION_CONTEXT_READY = True`. However, without a real external query task card, support-demo result/delta descriptors cannot be used as valid external metric evidence. Therefore, these policies are diagnostic-only (Diagnostic = ✅) and invalid for evidence (Valid = ❌).

The diagnostic oracle policies show the upper bound is **66.67%** (matching the P88 reproduction rate). All valid non-label evidence policies remain ceiling-bounded at 38.89% or 22.22%.

## 9. Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| p93_enriched_source_signature_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| task_context_absent_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| diagnostic_oracle_task_context_retrieval | 66.67% | 5.56% | 11.11% | +61.11% | +55.56% |
| diagnostic_support_delta_oracle_retrieval | 66.67% | 5.56% | 11.11% | +61.11% | +55.56% |
| non_label_external_task_manifest_retrieval | 22.22% | 11.11% | 11.11% | +11.11% | +11.11% |
| task_manifest_plus_source_similarity_retrieval | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |

Shuffled controls preserve true labels, permute vectors, and do not rotate target labels.

## 10. Leakage Audits

All non-label evidence policies pass leakage checks (`diagnostic_pass = True`). Diagnostic oracle policies are flagged as expected (114 occurrences of hidden labels and operators used), and they are excluded from evidence flags.

## 11. Proxy Leakage Audit

Proxy audit flags relation purity over task card buckets and descriptor hashes:
- `task_context_proxy_leakage_audit` defines and runs checks.
- Policy selections for improved policies show proxy risk (flagged), ensuring semantic geometry and bridge claims remain false.
- `proxy_leakage_risk_present` = True.

## 12. Evidence Verdict

No valid real external task context exists in the current synthetic dataset.

- `VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE` = **False**
- `SUPPORT_DEMONSTRATION_CONTEXT_READY` = **True**
- `TASK_CONTEXT_SIGNAL_PRESENT` = **False**
- `TASK_CONTEXT_EXTERNAL_METRIC_SUPPORTED` = **False**
- `EXTERNAL_TASK_CONTEXT_REQUIRED` = **True**

## 13. Recommended Next Phase

`P95_real_external_task_context_dataset_contract_no_training_no_bridge`

Because no valid real external task context exists in the current synthetic dataset, we must construct a dataset contract in P95 that provides a true non-label external task channel (e.g. prompt-side task definitions or demonstration-visible metadata) to guide retrieval.

## 14. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. learned_metric_evidence_not_present
2. semantic_metric_not_ready
3. real_external_task_context_dataset_not_available
4. external_context_metric_not_supported_for_bridge
5. bridge_input_contract_not_defined
6. bridge_validation_not_run

## 15. Limitations

- Without query task cards, support demo context descriptors are unguided and perform at chance (22.22%).
- Diagnostic oracle policies show the support-delta metric works (66.67%), but cannot be deployed without the hidden labels.

## 16. Final Verdict

**P94_READY_FOR_REVIEW**
