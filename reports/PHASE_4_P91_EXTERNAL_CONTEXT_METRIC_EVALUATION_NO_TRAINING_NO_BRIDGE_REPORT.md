# PHASE 4 / P91 — External Context Metric Evaluation Report

**NO TRAINING / NO TORCH / NO BRIDGE**

## Task

`PHASE_4_P91_EXTERNAL_CONTEXT_METRIC_EVALUATION_NO_TRAINING_NO_BRIDGE`

## Branch

`phase4/p91-external-context-metric-evaluation-no-training-no-bridge`

Based on accepted P90 head: `ce1992551a7daa03ca6b750b8eca1eede51aa2cc`

## Scientific Question

> When support is selected without hidden relation labels (non-label-selected policies),
> does the P88/P89-compatible 18-dimensional support-delta metric still produce
> classification signal above negative controls?

## Source Contracts

| Contract | Status |
|---|---|
| P90 phase/verdict validated | ✅ |
| P90 external_context_builder_ready | ✅ |
| P90 non_label_selected_support_ready | ✅ |
| P90 metric_shape_compatibility_pass | ✅ |
| P90 no training preserved | ✅ |
| P90 bridge not ready preserved | ✅ |
| P90 recommended P91 | ✅ |

## Policy Metric Results

| Policy | Eff. Accuracy | Coverage | Acc. on Supported | Train Ctx | Test Ctx | Supported | Unsupported | Diagnostic |
|---|---|---|---|---|---|---|---|---|
| label_selected_oracle_support | 66.67% | 100.00% | 66.67% | 76 | 18 | 18 | 0 | ✅ (diag only) |
| observable_domain_split_retrieval | 22.22% | 100.00% | 22.22% | 76 | 18 | 18 | 0 | ❌ |
| external_manifest_support | 22.22% | 100.00% | 22.22% | 76 | 18 | 18 | 0 | ❌ |

### Interpretation

- **Oracle (label-selected)** reproduces P88/P90 accuracy of 66.67%, confirming the metric works when the correct same-relation support is selected. This is diagnostic only and does not count toward non-label-selected evidence.
- **Observable domain split retrieval** achieves only 22.22%, matching chance-level for non-label-selected support.
- **External manifest support** achieves only 22.22%, also chance-level.

The metric signal collapses when support is not selected by hidden relation labels. This is an expected and honest result: the support-delta metric requires same-relation support to be informative.

## Negative Controls

| Policy | Normal | Shuffled Δ | Zero Δ | Beats Shuffled | Beats Zero |
|---|---|---|---|---|---|
| label_selected_oracle_support | 66.67% | 5.56% | 11.11% | +61.11% | +55.56% |
| observable_domain_split_retrieval | 22.22% | 11.11% | 11.11% | +11.11% | +11.11% |
| external_manifest_support | 22.22% | 11.11% | 11.11% | +11.11% | +11.11% |

### Control Properties

| Property | Value |
|---|---|
| shuffled_support_delta_control_implemented | ✅ true |
| shuffled_control_target_labels_preserved | ✅ true |
| shuffled_control_support_deltas_permuted | ✅ true |
| target_labels_shifted | ❌ false |
| true_labels_preserved | ✅ true |
| query_source_only_negative_control_applicable | ❌ false (delta-only metric) |

## Proxy Leakage Audit

| Observable Metadata Field | Unique Values | Max Relation Purity | Proxy Risk |
|---|---|---|---|
| observable_source_shape | 2 | 22.22% | ❌ No |
| observable_delta_shape | 3 | 22.64% | ❌ No |
| observable_context_hash | 6 | 25.00% | ❌ No |
| domain_split_source_shape | 6 | 25.00% | ❌ No |
| domain_split_delta_shape | 7 | 25.00% | ❌ No |

**Result:** No observable metadata field leaks relation labels (all purities < 95% threshold).

## Leakage Verification (Non-Label-Selected Policies)

| Policy | Hidden Label Used | Query Target Used | Query Delta Used | Operator ID Used | Pass |
|---|---|---|---|---|---|
| observable_domain_split_retrieval | 0 | 0 | 0 | 0 | ✅ |
| external_manifest_support | 0 | 0 | 0 | 0 | ✅ |

## Shape Compatibility

| Policy | Compatible | Incompatible | All Compatible | Pass |
|---|---|---|---|---|
| observable_domain_split_retrieval | 114 | 0 | ✅ | ✅ |
| external_manifest_support | 114 | 0 | ✅ | ✅ |

## Evidence Flags

### Non-Label-Selected Metric Signal Criteria

For signal to be present, a non-label-selected policy must satisfy ALL:
- effective_accuracy >= 40%
- coverage >= 80%
- beats shuffled control by >= 10%
- beats zero-delta control by >= 10%
- leakage clean
- shape compatible

| Policy | Eff. Acc ≥ 40% | Cov ≥ 80% | Beats Shuf ≥ 10% | Beats Zero ≥ 10% | Leak Clean | Shape OK | Pass |
|---|---|---|---|---|---|---|---|
| observable_domain_split_retrieval | ❌ 22.22% | ✅ 100% | ✅ 11.11% | ✅ 11.11% | ✅ | ✅ | ❌ |
| external_manifest_support | ❌ 22.22% | ✅ 100% | ✅ 11.11% | ✅ 11.11% | ✅ | ✅ | ❌ |

**NON_LABEL_SELECTED_METRIC_SIGNAL_PRESENT = False**

Both non-label-selected policies fail the accuracy threshold (22.22% < 40%).

**EXTERNAL_CONTEXT_METRIC_SUPPORTED = False**

Cannot be true when no non-label-selected metric signal is present.

### Gate Flags

| Flag | Value |
|---|---|
| nonlearned_metric_signal_present | ✅ True (from P88) |
| hard_generalization_supported | ✅ True (from P89) |
| external_context_builder_ready | ✅ True (from P90) |
| non_label_selected_support_ready | ✅ True (from P90) |
| non_label_selected_metric_signal_present | ❌ False |
| external_context_metric_supported | ❌ False |
| learned_selector_evidence_present | ❌ False |
| learned_metric_evidence_present | ❌ False |
| semantic_metric_ready | ❌ False |
| bridge_implementation_allowed | ❌ False |
| bridge_ready | ❌ False |
| generation_claims_allowed | ❌ False |
| semantic_geometry_claims_allowed | ❌ False |

## Bridge Boundary

Bridge remains blocked. Blocking reasons:
1. learned_metric_evidence_not_present
2. semantic_metric_not_ready
3. external_metric_not_sufficient_for_bridge
4. bridge_input_contract_not_defined
5. bridge_validation_not_run

## Constraints

| Constraint | Status |
|---|---|
| No torch import | ✅ |
| No numpy/pandas/sklearn/random import | ✅ |
| No model training | ✅ |
| No optimizer | ✅ |
| No checkpoint | ✅ |
| No artifact written | ✅ |
| Exactly 5 planned files | ✅ |
| Oracle diagnostic only | ✅ |
| P69–P90 files unmodified | ✅ |

## Recommended Next Phase

`P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge`

The non-label-selected support policies do not provide same-relation support, so the metric collapses. Next phase should explore improved support retrieval strategies that can approximate same-relation matching without using hidden labels.

## Verdict

**P91_READY_FOR_REVIEW**

## Files

| # | File | Status |
|---|---|---|
| 1 | `src/phase4/external_context_metric_evaluation.py` | NEW |
| 2 | `tools/phase4/run_p91_external_context_metric_evaluation_smoke.py` | NEW |
| 3 | `tests/test_phase4_external_context_metric_evaluation.py` | NEW |
| 4 | `tests/test_phase4_p91_external_context_metric_evaluation_smoke.py` | NEW |
| 5 | `reports/PHASE_4_P91_EXTERNAL_CONTEXT_METRIC_EVALUATION_NO_TRAINING_NO_BRIDGE_REPORT.md` | NEW |

## Test Results

26 passed, 0 failed.
