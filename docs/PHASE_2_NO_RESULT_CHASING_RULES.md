# PHASE 2 NO-RESULT-CHASING RULES
# Prohibition List and Claim Language Rules

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — these rules are binding from the moment this document is committed

> [!CAUTION]
> **These rules are active from the moment this commit is merged or acknowledged by the project owner.**
> Violation of any rule in this document invalidates the associated experiment run.
> An invalidated run may not be cited as evidence for any claim.

---

## 1. Metrics Must Be Frozen Before Training

No metric definition, validity check, or computational procedure may be created or modified
after any Phase 2 training run begins.

**What "frozen" means:**
- The metric definition is committed to `docs/PHASE_2_METRIC_CONTRACT.md`
- The commit hash of that document is recorded in the run manifest
- The exact metric computation code is committed before training starts
- No change to metric computation is permitted post-hoc

**Violation example:** Changing from Euclidean distance to Mahalanobis distance for
`generated_C_novelty_score` after seeing that Euclidean distance gives a low score.
RESULT: All runs computed with Euclidean must remain Euclidean; new runs must be started fresh
with the new metric under a new experiment ID.

---

## 2. Thresholds Must Be Frozen Before Training

All numeric thresholds in `docs/PHASE_2_METRIC_CONTRACT.md` must be resolved from
`TO_BE_DEFINED` to specific numeric values and committed before any Phase 2 training begins.

**What "frozen" means:**
- Every TO_BE_DEFINED item in `docs/PHASE_2_METRIC_CONTRACT.md` is replaced with a specific value
- The document is committed and the commit hash is recorded
- The same commit hash appears in `metric_contract_snapshot.md` in every run artifact

**Violation example:** Setting `generated_C_valid_rate_threshold = 30%` after seeing that the
constrained model achieves 40% but the unconstrained model achieves 35%. RESULT: Run invalid.

**Allowed:** Setting `generated_C_valid_rate_threshold = 50%` before any model is trained, based on
theoretical analysis of the ARMA-GARCH parameter space validity rate. Then observing that both
models fail. This is a valid negative result, not a violation.

---

## 3. Family Definitions Must Be Frozen Before Data Generation

The exact mathematical definition of families A, B, and C — including all boundary cases —
must be committed to `docs/PHASE_2_SPLIT_CONTRACT.md` before any dataset is generated.

**Violation example:** Deciding after generation that AR models with q=0 are "not really ARMA"
and should be excluded from A-Train, then regenerating the dataset. RESULT: Both datasets must
be archived; the boundary case decision must be committed; then new data generated with a new seed.

---

## 4. Split Contract Must Be Frozen Before Data Generation

The exact split contract (which samples are in A-Train, B-Train, C-Holdout) must be committed
and hashed before any data generation begins.

**Violation example:** Moving samples between C-Holdout and A-Train after seeing that C-Holdout
validity rate is low. RESULT: Split violation; the run is invalid.

---

## 5. Negative Results Must Be Preserved

Any experiment run that produces results below the precommitted thresholds must be preserved as-is.

**What "preserved" means:**
- All artifacts are kept (not deleted)
- The verdict.json records `"run_outcome": "FAIL_..."` and `"negative_result": true`
- The failure is documented in the Phase 2 final report
- The run is not repeated with different settings without creating a new experiment ID and documenting the change

**Violation example:** Deleting a run directory because the generated_C_valid_rate was 0%.
RESULT: Research integrity violation.

**Correct action:** Archive the run, document the failure mode in `failure_cases.csv` and
`report.md`, and start a new experiment with documented changes.

---

## 6. No Post-Hoc Metric Changes

Once a metric is computed for a run, its value may not be recomputed or updated unless:
- A verified bug in the metric computation code is found and documented
- The bug fix is committed with an explicit note identifying the bug
- The run is re-executed from scratch with the fixed code under a new run_id
- The original run artifact is preserved with a note pointing to the corrected run

Post-hoc recomputation without these conditions is forbidden.

---

## 7. No Post-Hoc Threshold Changes

Once the metric contract is approved and training has begun:
- No threshold value may be changed
- No new threshold may be added that would reclassify a failing run as passing
- No threshold may be removed

If a threshold proves scientifically unjustifiable after experiments (not after seeing results),
a new metric contract version may be created for future experiments. All past runs remain
evaluated under the contract version recorded in their manifest.

---

## 8. No Deleting Failed Runs

Failed experiment runs must be preserved. The following actions are unconditionally prohibited:
- Deleting a run directory
- Deleting individual artifact files from a completed run
- Overwriting `verdict.json` after a run completes
- Overwriting `metrics.json` after a run completes
- Moving a run directory to a location not tracked by the artifact contract

**Disk space management:** If disk space is a concern, artifact binaries (generated_samples.parquet,
environment_freeze.txt) may be compressed, but they may not be deleted. The manifest and metrics
files must always remain intact.

---

## 9. No Overwriting Artifacts

Once a run produces an artifact file, that file is immutable. The following are forbidden:
- Rewriting `manifest.json` after the run completes
- Updating `metrics.json` with additional metrics computed post-hoc
- Replacing `generated_samples.parquet` with a filtered or corrected version
- Replacing `config_snapshot.yaml` with a corrected config

If an artifact is found to be incorrect, the correct procedure is:
1. Document the error in an `errata.md` file in the run directory
2. Start a new run with the corrected code/config under a new run_id
3. Reference the original run_id in the new run's manifest under `"corrects_run": "<original_run_id>"`

---

## 10. No Claim Without Evidence

No claim about Phase 2 experimental results may be made without:
1. A completed run with a full artifact archive
2. A `verdict.json` with `"claim_authorized": true`
3. A `manifest.json` with all required fields present and non-null
4. A split manifest verifying `C_count_in_zero_shot_train == 0`

**Forbidden claims without evidence:**
- "The model can generate ARMA-GARCH processes" — requires evidence per §11
- "The latent space encodes compositional structure" — requires evidence per §11
- "Phase 2 confirms zero-shot generalization" — requires evidence per §11
- "The model outperforms the baseline" — requires a completed baseline run under identical conditions

---

## 11. Required Conditions Before "Model Generated C" Claim

All seven conditions must be met simultaneously. Meeting five of seven is insufficient.

### Condition 1 — Zero-Shot Training Verified
- `C_count_in_zero_shot_train == 0` in `split_manifest.json`
- Split manifest hash in `manifest.json` matches precommitted hash

### Condition 2 — Validity Rate Above Threshold
- `generated_C_valid_rate` ≥ APPROVED_PROTOCOL_THRESHOLD as recorded in `metrics.json`
- The threshold value matches `thresholds_used.generated_C_valid_rate_threshold` in `verdict.json`
- `verdict.json` shows `"per_metric_verdict.generated_C_valid_rate": "PASS"`

### Condition 3 — Composition Score Above Threshold
- `generated_C_composition_score` ≥ APPROVED_PROTOCOL_THRESHOLD
- This confirms the generated C is a genuine ARMA-GARCH composition, not a degenerate trivial case

### Condition 4 — Novelty Score Above Threshold
- `generated_C_novelty_score` ≥ APPROVED_PROTOCOL_THRESHOLD
- This confirms the generated C is not memorization of training samples

### Condition 5 — Seed Stability
- Result must be replicated across ≥ N independent seeds (N: APPROVED value from metric contract)
- `seed_stability_range` ≤ APPROVED_PROTOCOL_THRESHOLD
- Each seed run must have its own complete artifact archive

### Condition 6 — Unconstrained Baseline Compared
- At least one unconstrained baseline (B1 — standard VAE) must have been evaluated under identical conditions
- The baseline run must have its own complete artifact archive
- The comparison must be documented in the final report

### Condition 7 — Full Artifact Archive Exists
- `artifact_completeness_score == 1.0` in `manifest.json`
- All required files in §1 must be present and non-empty
- `git_dirty` must be `false` in `manifest.json`

---

## 12. Forbidden Claim Examples (Detailed)

### "The model discovered ARMA-GARCH structure in the latent space."
**Forbidden because:** "Discovered" implies automatic learning without verification. The claim is too vague and cannot be falsified. The latent space geometry must be explicitly characterized against the metric contract.

### "The latent manifold encodes compositional structure."
**Forbidden because:** This is a geometric claim about the VAE manifold. It requires explicit geometric evidence (curvature, geodesics, Jacobian analysis) that is separate from the C-validity metrics. Neither PCA visualizations nor UMAP plots constitute evidence for this claim.

### "Our model generates valid hybrid time series."
**Forbidden because:** "Valid" must be quantified against the metric contract. "Hybrid" must mean ARMA-GARCH as defined in the split contract. "Time series" must specify the simulation procedure. The claim must include: validity rate, confidence interval, number of seeds, comparison baseline, and experiment type (zero-shot/few-shot/supervised).

### "Phase 2 confirms zero-shot C generalization."
**Forbidden because:** This is a summary claim that may only appear in the final report after conditions 1-7 in §11 are all met and documented.

### "The model learns to compose families."
**Forbidden because:** "Learns" is undefined. "Compose" requires composition score above threshold. "Families" must be explicitly defined as A and B in the split contract.

### "The latent space has a smooth composition manifold."
**Forbidden because:** Manifold smoothness (geodesic smoothness, curvature) is a geometric property that requires explicit geometric metrics. It does not imply structural validity of decoded points. Latent interpolation smoothness is not H1 evidence (see H2).

---

## 13. Allowed Claim Language

The following are templates for valid claim language. They require the indicated evidence.

### Template T1 — Zero-Shot Validity Rate Claim
```
"Under the zero-shot C protocol (C_count_in_train=0, verified by split manifest hash <hash>),
the <model_name> model achieved generated_C_valid_rate = <X>% (threshold: <T>%) across
<N> independent seeds (range: [<lo>%, <hi>%])."
```

### Template T2 — Comparison Claim
```
"The constrained model (<name>) achieved generated_C_valid_rate = <X>% compared to
the unconstrained VAE baseline at <Y>%, both evaluated under identical zero-shot conditions
using the same data split (manifest hash <hash>)."
```

### Template T3 — Negative Result Claim
```
"No model evaluated under the zero-shot C protocol achieved generated_C_valid_rate above
the precommitted threshold of <T>%. This confirms H0: unconstrained VAE models trained on
A ∪ B do not reliably generate valid ARMA-GARCH instances without structural inductive bias.
[If constrained models also fail: The constrained variants also failed to exceed the threshold;
this suggests that the A+B→C composition rule requires <additional inductive bias not tested>.]"
```

### Template T4 — Partial Evidence Claim
```
"The model achieved generated_C_valid_rate = <X>% (below the precommitted threshold of <T>%).
This result does not authorize a zero-shot C claim. The failure mode analysis in
failure_cases.csv indicates <dominant failure mode>."
```
