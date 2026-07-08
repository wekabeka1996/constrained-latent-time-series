# PHASE 4 / P97 — Task Card Retrieval Policy Repair Report

**TYPED OPERATOR-INTENT GRAPH / NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P97_TASK_CARD_RETRIEVAL_POLICY_REPAIR_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p97-task-card-retrieval-policy-repair-no-training-no-bridge`

Based on P96 branch head: `be7712a3f3e16c1ad07b81352cdcdb4e2e3d9af8`

---

## 1. Problem Framing

P96 showed that primitive public token overlap retrieval (27.78%) falls short of selecting high-quality support demonstrations because Jaccard word matching cannot represent the semantic alignment of transformation operations. P97 repairs retrieval policy logic by building a deterministic typed operator-intent graph over public fields of task cards and support demonstrations.

## 2. P96 Inheritance

P97 validates and inherits the P96 contract state:
- `task_card_retrieval_pipeline_ready` = True
- `task_card_retrieval_signal_present` = False
- `diagnostic_task_card_id_exact_match_effective_accuracy` = 66.67%
- `p93_enriched_source_signature_baseline_effective_accuracy` = 38.89%
- `valid_external_task_context_available` = False
- `proxy_risk_present` = True

## 3. No-Training / No-Torch Boundary

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer created | ✅ |
| No checkpoint written | ✅ |

P97 does not train a model. P97 does not prove learned metric evidence. P97 does not prove semantic geometry. P97 does not validate any bridge. P97 repairs retrieval policy logic using deterministic typed public task-card structure. Any signal remains diagnostic-only because P95/P96 task-card assignments are label-mapped.

## 4. Why Primitive Token Retrieval Failed

Token overlap (Jaccard) simply counts common strings. For example, if a query instruction says "Shift sequence vertically" and a support instruction says "Shift sequence horizontally", they share "Shift", "sequence", "axis", and "along" producing high Jaccard similarity despite requesting orthogonal changes. Structured slots are needed to capture these distinctions.

## 5. Typed Operator-Intent Graph

Task cards are parsed into structured signatures:
- `intent_family` (translation, scaling, reflection, frequency/phase shift, volatility/trend shift)
- `change_axis_type` (horizontal coordinate, vertical value, sequence length, temporal phase, frequency, amplitude, etc.)
- `change_direction_type` (increase, decrease, invert, shift, compress/stretch)
- `geometry_action_type`, `temporal_action_type`, `value_action_type`
- `locality_type` (global, local, envelope)
- `periodicity_type` (periodic, non_periodic)
- `preserve_vector` & `forbid_vector` (constraints)

Matching is done via a deterministic weighted **typed slot distance**.

## 6. Support Demo Visible Transformation Signature

Support demonstrations are parsed into structured signatures containing:
- `demo_change_scale_type` ("large", "medium", "small" based on delta magnitude)
- `demo_dimension_type` (e.g. "dim_0" based on changed dimensions)
- `demo_preservation_type`
- `demo_caption_intent_signature` (extracted from the human demo caption)
- `demo_task_intent_signature` (extracted from the demo task card)

## 7. Repaired Retrieval Policies

We evaluate 14 policies including baselines, diagnostic controls, structured public policies, and ablations. Key repaired public policies are:
1. `typed_operator_intent_signature_retrieval`
2. `typed_intent_plus_preserve_constraint_retrieval`
3. `demo_visible_transformation_signature_retrieval`
4. `typed_intent_demo_compatibility_retrieval`
5. `hybrid_typed_intent_demo_source_retrieval`

## 8. Metric Results

| Policy | Effective Accuracy | Coverage | Supported Accuracy | Diagnostic | Valid | Pilot |
|---|---|---|---|---|---|---|
| p93_enriched_source_signature_baseline | 38.89% | 100.00% | 38.89% | ❌ | ❌ | ✅ |
| task_card_absent_baseline | 38.89% | 100.00% | 38.89% | ❌ | ❌ | ✅ |
| p96_public_token_overlap_baseline | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| p96_change_preserve_baseline | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| **diagnostic_task_card_id_exact_match** | **66.67%** | 100.00% | 66.67% | ✅ | ❌ | ❌ |
| **diagnostic_compatible_label_mapping_oracle** | **66.67%** | 100.00% | 66.67% | ✅ | ❌ | ❌ |
| typed_operator_intent_signature_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| typed_intent_plus_preserve_constraint_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| demo_visible_transformation_signature_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| typed_intent_demo_compatibility_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| hybrid_typed_intent_demo_source_retrieval | 11.11% | 100.00% | 11.11% | 18 | 18 | 0 |
| preserve_constraints_only_retrieval | 38.89% | 100.00% | 38.89% | ❌ | ❌ | ✅ |
| change_axis_only_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |
| demo_caption_intent_only_retrieval | 27.78% | 100.00% | 27.78% | ❌ | ❌ | ✅ |

The diagnostic oracle policies show that exact card-id matches achieve **66.67%**. Public structured retrieval policies achieve **27.78%** accuracy, which does not cross the 40.00% threshold.

## 9. Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| typed_operator_intent_signature_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |
| typed_intent_plus_preserve_constraint_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |
| demo_visible_transformation_signature_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |
| typed_intent_demo_compatibility_retrieval | 27.78% | 5.56% | 11.11% | +22.22% | +16.67% |

## 10. Typed Signature Shuffle Controls

The typed signature shuffle control permutes task card signatures across queries while preserving true labels and support demonstrations:
- `typed_operator_intent_signature_retrieval`: shuffled signature accuracy = **11.11%** (degradation = **16.67%**).
- `typed_intent_plus_preserve_constraint_retrieval`: shuffled signature accuracy = **11.11%** (degradation = **16.67%**).

This degradation confirms that the retrieval policy is actively utilizing the structured task card signatures to select demonstrations.

## 11. Leakage Audits

All public policies pass leakage checks (`diagnostic_pass = True`). No public policy leaks `task_card_id` or `compatible_relation_labels`.

## 12. Proxy Audit

Proxy risk remains present (`proxy_risk_present = True`) because P95 task card assignments are synthetically label-mapped.

## 13. Diagnostic Repaired Signal

`REPAIRED_TASK_CARD_RETRIEVAL_SIGNAL_PRESENT` = **False**

No public-token policy crossed the 40.00% signal threshold.

## 14. Real Evidence Boundary

- `VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE` = **False**
- `VALID_FOR_REAL_EXTERNAL_TASK_CONTEXT_EVIDENCE` = **False**

Even structured public policies remain invalid for real evidence due to synthetic mapping.

## 15. Recommended Next Phase

`P98_task_card_schema_enrichment_or_controlled_real_data_seed_no_training_no_bridge`

Because even structured public fields (27.78%) are insufficient to beat the source baseline (38.89%), the system requires either schema enrichment (such as adding magnitude bounds to task cards) or a controlled real-data seed.

## 16. Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. p97_is_diagnostic_retrieval_repair_not_real_external_evidence
2. task_card_assignments_are_label_mapped_from_synthetic_data
3. valid_external_task_context_evidence_not_available
4. learned_metric_evidence_not_present
5. semantic_metric_not_ready
6. bridge_input_contract_not_defined
7. bridge_validation_not_run

## 17. Limitations

- A rule-based parser over synthetic card text is deterministic but lacks semantic flexibility for open-domain descriptions.
- The retrieval accuracy ceiling is heavily constrained by the quality and uniqueness of public descriptors.

## 18. Final Verdict

**P97_READY_FOR_REVIEW**
