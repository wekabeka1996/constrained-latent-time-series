# PHASE 4 / P96 — Real Task Card Support Retrieval Pilot Report

**REAL-ISH PILOT / NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P96_REAL_TASK_CARD_SUPPORT_RETRIEVAL_PILOT_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p96-real-task-card-support-retrieval-pilot-no-training-no-bridge`

Based on P95 branch head: `decf3d94895fd9ecc2ed1e92fb9870ad25701298`

---

## 1. Problem Framing

P95 defined the real external task context dataset contract mapping relation identity to label-free task cards. P96 pilots this contract by executing retrieval policies that match query task cards against support demo task cards using deterministic distance functions (Jaccard word overlap, preserve constraint mismatch, support caption matching).

## 2. P95 Inheritance

P96 validates and inherits the P95 contract state:
- `REAL_EXTERNAL_TASK_DATASET_CONTRACT_READY` = True
- `TASK_CARD_RECORDS_BUILT` = True
- `QUERY_TASK_CARD_RECORDS_BUILT` = True
- `SUPPORT_DEMONSTRATION_CARD_RECORDS_BUILT` = True
- `VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE` = False (diagnostic mapping only)
- `ready_for_real_data_collection` = True

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |

P96 does not train a model. P96 does not prove learned metric evidence. P96 does not prove semantic geometry. P96 does not validate any bridge. P96 is a diagnostic pipeline pilot because P95 task-card assignments are label-mapped. Even public-token retrieval inherits assignment proxy risk until real raw task-card data is collected before labels.

## 4. Pilot Policies

We evaluate 8 retrieval policies:

### Baselines
1. `p93_enriched_source_signature_baseline`: Enriched source similarity.
2. `task_card_absent_baseline`: No task card; same-domain/same-split fallback.

### Diagnostic Upper Bound (Diagnostic-Only)
3. `diagnostic_task_card_id_exact_match`: Filter matching exact `task_card_id`.
4. `diagnostic_compatible_label_mapping_oracle`: Reconstruct oracle using relation labels.

### Public policies
5. `task_card_public_token_overlap_retrieval`: Overlap between query task card public tokens and support card public tokens.
6. `task_card_change_preserve_signature_retrieval`: Overlap of change intents combined with preserve constraints Hamming distance.
7. `support_demo_public_caption_retrieval`: Overlap between query public tokens and support human captions.
8. `hybrid_task_card_plus_source_signature_retrieval`: Combines public card distance with source signature distance.

## 5. Public Task-Card Distance Functions

- **Jaccard overlap:** `1.0 - (intersection / union)` over lowercased, stopword-filtered token lists.
- **Preserve constraint distance:** Mismatch ratio (Hamming distance) over boolean preserve constraints.
- **Task card public distance:** Weighted combination of primary change, change axis description, and preserve constraints distances.

No policy in this category utilizes `task_card_id` or `compatible_relation_labels` for distance calculation.

## 6. Support-Demo Public-View Distance

Computes distance between query public tokens and support caption/descriptor fields without utilizing hidden labels.

## 7. Metric Results

| Policy | Effective Accuracy | Coverage | Supported Accuracy | Test Ctx | Supported | Unsupported | Diagnostic | Valid | Pilot |
|---|---|---|---|---|---|---|---|---|---|
| p93_enriched_source_signature_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ | ❌ | ✅ |
| task_card_absent_baseline | 38.89% | 100.00% | 38.89% | 18 | 18 | 0 | ❌ | ❌ | ✅ |
| **diagnostic_task_card_id_exact_match** | **66.67%** | 100.00% | 66.67% | 18 | 18 | 0 | ✅ | ❌ | ❌ |
| **diagnostic_compatible_label_mapping_oracle** | **66.67%** | 100.00% | 66.67% | 18 | 18 | 0 | ✅ | ❌ | ❌ |
| task_card_public_token_overlap_retrieval | 27.78% | 100.00% | 27.78% | 18 | 18 | 0 | ❌ | ❌ | ✅ |
| task_card_change_preserve_signature_retrieval | 27.78% | 100.00% | 27.78% | 18 | 18 | 0 | ❌ | ❌ | ✅ |
| support_demo_public_caption_retrieval | 27.78% | 100.00% | 27.78% | 18 | 18 | 0 | ❌ | ❌ | ✅ |
| hybrid_task_card_plus_source_signature_retrieval | 11.11% | 100.00% | 11.11% | 18 | 18 | 0 | ❌ | ❌ | ✅ |

The diagnostic oracle policies show that exact card-id alignment matches the P88 reproduction rate of **66.67%**. Valid public-token policies do not cross the 40.00% signal threshold.

## 8. Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| p93_enriched_source_signature_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| task_card_absent_baseline | 38.89% | 0.00% | 11.11% | +38.89% | +27.78% |
| diagnostic_task_card_id_exact_match | 66.67% | 5.56% | 11.11% | +61.11% | +55.56% |
| diagnostic_compatible_label_mapping_oracle | 66.67% | 5.56% | 11.11% | +61.11% | +55.56% |
| task_card_public_token_overlap_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |
| task_card_change_preserve_signature_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |
| support_demo_public_caption_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |

Shuffled controls permute vectors, preserve labels, and do not rotate target labels.

## 9. Leakage Audits

All public retrieval policies pass leakage checks (`diagnostic_pass = True`). No public policy utilizes `task_card_id` or `compatible_relation_labels`.

## 10. Proxy Audit

Proxy risk is present (`proxy_risk_present = True`) because P95 card assignments are label-mapped from existing synthetic data.

## 11. Diagnostic Pipeline Signal

`TASK_CARD_RETRIEVAL_SIGNAL_PRESENT` = **False**

None of the public-token policies crossed the 40.00% signal threshold.

## 12. Real Evidence Boundary

- `VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE` = **False**
- `VALID_FOR_REAL_EXTERNAL_TASK_CONTEXT_EVIDENCE` = **False**

Even public policies are invalid for real evidence because query assignments are synthetically label-mapped.

## 13. Recommended Next Phase

`P97_task_card_retrieval_policy_repair_no_training_no_bridge`

Because the diagnostic pipeline contract exists but public retrieval policies fail to cross the 40.00% signal threshold, the retrieval policy logic must be repaired in P97 (e.g. by using refined Jaccard weights or soft description token embeddings) to exceed the P93 source baseline.

## 14. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. p96_is_diagnostic_pipeline_pilot_not_real_external_evidence
2. task_card_assignments_are_label_mapped_from_synthetic_data
3. valid_external_task_context_evidence_not_available
4. learned_metric_evidence_not_present
5. semantic_metric_not_ready
6. bridge_input_contract_not_defined
7. bridge_validation_not_run

## 15. Limitations

- Retrieval over simple public descriptor word lists is underdetermined for semantic similarity.
- Synthetic label-mapping introduces a strong proxy dependency that can only be cleared with raw real-world prompt task cards.

## 16. Final Verdict

**P96_READY_FOR_REVIEW**
