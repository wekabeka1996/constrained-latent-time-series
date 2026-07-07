# PHASE 4 / P95 — Real External Task Context Dataset Contract Report

**REAL-ISH TASK CARDS / SUPPORT DEMOS / NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P95_REAL_EXTERNAL_TASK_CONTEXT_DATASET_CONTRACT_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p95-real-external-task-context-dataset-contract-no-training-no-bridge`

Based on accepted repaired P94 head: `902ec0b9b6256c2ac2f2b5fa28d077bed6f593e4`

---

## 1. Problem Framing

P94 proved that support retrieval cannot break the source-similarity ceiling of 38.89% without access to task-level description because the mapping is underdetermined from source observations alone. P95 designs the dataset contract to introduce task cards as a public task channel (e.g. human-readable instructions, change intent, and preserve constraints) rather than relying on hidden relation labels.

## 2. P94 Inheritance

P95 validates and inherits the P94 diagnostic results:
- `diagnostic_oracle_task_context_retrieval` = 66.67%
- `diagnostic_support_delta_oracle_retrieval` = 66.67%
- `p93_enriched_source_signature_baseline` = 38.89%
- `task_context_absent_baseline` = 38.89%
- `support_demo_context_without_query_task_card` and `support_demo_context_with_non_label_task_manifest` are correctly set as diagnostic-only and invalid for evidence.
- No valid external task context existed in P94.

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |

P95 does not train a model. P95 does not prove learned metric evidence. P95 does not prove semantic geometry. P95 does not validate any bridge. P95 defines the real external task context dataset contract only. Current synthetic assignments are diagnostic-only and label-mapped. Real external task context evidence requires raw records with task cards provided before labels.

## 4. Why Task Cards Are Needed

Without external task cards describing the intended transformation, a retrieval selector cannot know which aspects of the query source coordinate change to match against (e.g. amplitude vs translation vs frequency). Task cards formalize intent in a label-free manner, specifying change intents and preservation constraints.

## 5. Task Card Schema

Task cards contain:
- `task_card_id` & `task_card_version`
- `human_readable_instruction`
- `change_intent` (primary_change, secondary_change, change_axis_description)
- `preserve_constraints` (e.g. `preserve_sequence_length`, `preserve_temporal_order`, etc.)
- `forbidden_changes` (relation_label_leak, operator_id_leak, query_target_leak, query_delta_leak)
- `public_descriptor_tokens`
- `task_card_audit` (uses_hidden_relation_label = False, valid_external_task_context = True)
- `audit_label_mapping_only` (compatible_relation_labels)

## 6. Query Task Card Record Schema

Query records are mapped to task cards diagnostically in the synthetic setup:
- `query_task_card_record_id`
- `query_task_card`
- `assignment_source` = "diagnostic_contract_mapping_from_existing_synthetic_label"
- `assignment_audit` (uses_hidden_relation_label_for_assignment = True, valid_for_real_external_task_context_evidence = False, valid_for_dataset_contract_validation = True, diagnostic_only = True)

## 7. Support Demonstration Card Schema

Support demonstration cards match the query task card schema structure:
- `support_demo_public_view` (source_observation, result_observation, visible_change_summary, visible_preservation_summary, human_demo_caption)
- `support_demo_card_audit` (uses_hidden_relation_label_for_assignment = True, valid_for_real_external_task_context_evidence = False, valid_for_dataset_contract_validation = True, diagnostic_only = True)

## 8. Dataset Contract Public/Hidden Fields

| Public to Retrieval | Hidden from Retrieval |
|---|---|
| Query source observation | `target_relation_label` |
| Query task card | `operator_id` |
| Support source observation | `query target` |
| Support result observation | `query delta` |
| Support visible change/preservation summaries | `hidden relation family` |
| Support task card | `evaluation target B` |

## 9. Leakage Audit

Leakage check validates that public fields have no leakage of labels, operator IDs, or targets:
- `retrieval_public_fields_include_label_count` = 0
- `retrieval_public_fields_include_operator_id_count` = 0
- `contract_diagnostic_only` = True
- `valid_for_real_external_task_context_evidence` = False
- `valid_for_dataset_contract_validation` = True
- `diagnostic_pass` = True

## 10. Proxy Audit

Proxy audit verifies that synthetic assignments are label-mapped:
- `proxy_risk_present` = True
- `high_proxy_risk_fields` = `["task_card_id", "human_readable_instruction", "public_descriptor_tokens", "compatible_relation_labels"]`
- `assignment_is_diagnostic_label_mapped` = True
- `real_data_required_to_remove_assignment_proxy` = True

## 11. Evidence Boundary

- `REAL_EXTERNAL_TASK_DATASET_CONTRACT_READY` = **True**
- `TASK_CARD_RECORDS_BUILT` = **True**
- `QUERY_TASK_CARD_RECORDS_BUILT` = **True**
- `SUPPORT_DEMONSTRATION_CARD_RECORDS_BUILT` = **True**
- `VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE` = **False** (due to diagnostic-only synthetic label mapping)

## 12. Ready-for-Real-Data-Collection Verdict

`ready_for_real_data_collection` = **True**

The schemas, public/hidden field gates, and invariants are fully defined. The dataset contract is ready for integration with real-world task definition collection pipelines.

## 13. Recommended Next Phase

`P96_real_task_card_support_retrieval_pilot_no_training_no_bridge`

P96 will use this dataset contract to pilot a small-scale retrieval agent matching query task card public tokens against support demo card descriptors.

## 14. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. real_task_context_contract_only_no_real_data_yet
2. valid_external_task_context_evidence_not_available
3. learned_metric_evidence_not_present
4. semantic_metric_not_ready
5. bridge_input_contract_not_defined
6. bridge_validation_not_run

## 15. Limitations

- Synthetic data does not contain real-world prompt cards, so the current assignment maps existing labels to cards diagnostically, creating a high proxy risk. Real evidence requires raw task card definitions collected upfront.

## 16. Final Verdict

**P95_READY_FOR_REVIEW**
