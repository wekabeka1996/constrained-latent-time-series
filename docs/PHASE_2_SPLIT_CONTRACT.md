# PHASE 2 SPLIT CONTRACT
# A/B/C Family Split Rules, Zero-Shot / Few-Shot Protocols, Leakage Guard Requirements

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval before any data generation begins

> [!IMPORTANT]
> **This document must be reviewed and marked APPROVED before any data generation begins.**
> No training dataset, evaluation dataset, or holdout set may be created until this split
> contract is approved and all TO_BE_DEFINED items are resolved.

---

## 1. Exact Definitions of A-Train, B-Train, and C-Holdout

### 1.1 A-Train
- **Family:** AR / ARMA — mean dynamics only
- **Mathematical constraint:** Each sample must satisfy:
  - AR stationarity: all roots of the AR polynomial strictly inside the unit circle
  - MA invertibility: all roots of the MA polynomial strictly outside the unit circle (for ARMA)
  - Volatility component: absent (omega=0, alpha=[], beta=[])
- **Generator:** Phase 2 generator (to be implemented; `src/data_generator.py` in `clean_validated` mode is a candidate)
- **Sample count:** TO_BE_DEFINED — must be committed before data generation
- **Seed:** TO_BE_DEFINED — must be explicit, not `seed=42` silently inherited
- **Validity rate target:** 100% of generated samples must pass validity checks before training
  (if any sample fails, the generator must fail-fast and not silently exclude the sample)
- **Schema:** Phase 2 schema v2 (after schema spec approval)

### 1.2 B-Train
- **Family:** GARCH — volatility dynamics only
- **Mathematical constraint:** Each sample must satisfy:
  - ω > 0 (GARCH intercept strictly positive)
  - αᵢ ≥ 0 for all i (ARCH coefficients non-negative)
  - βⱼ ≥ 0 for all j (GARCH coefficients non-negative)
  - Σα + Σβ < 1 (persistence strictly below 1)
  - Mean component: absent (phi=[], theta=[])
- **Generator:** Phase 2 generator (must NOT use `legacy_compatible` mode — the Phase 1 GARCH beta bug must not appear in training data)
- **Sample count:** TO_BE_DEFINED — must be committed before data generation
- **Seed:** TO_BE_DEFINED — separate from A-Train seed; must be explicit
- **Validity rate target:** 100% of generated samples must pass validity checks

### 1.3 C-Holdout
- **Family:** ARMA-GARCH — combined mean + volatility hybrid
- **Mathematical constraint:** Each sample must simultaneously satisfy:
  - All A-family constraints (stationarity, invertibility)
  - All B-family constraints (non-negativity, persistence < 1)
- **Generator:** Phase 2 C generator (to be implemented as a separate function; must be completely separate from A and B generators)
- **Sample count:** TO_BE_DEFINED — must be committed before data generation
- **Seed:** TO_BE_DEFINED — separate from A and B seeds; must be explicit
- **Validity rate target:** 100% of generated samples must pass validity checks
- **Critical rule:** C-Holdout samples must NEVER appear in A-Train or B-Train under any protocol variant

---

## 2. Zero-Shot Rule

The zero-shot protocol has a single inviolable rule:

> **No C sample, no C label, no C ModelSpec, and no ARMA-GARCH mean+volatility hybrid may appear
> in the training dataset under the zero-shot protocol.**

### 2.1 What This Means Concretely
- A-Train contains only samples where `volatility_family = "none"`
- B-Train contains only samples where `mean_family = "none"`
- No sample where both `mean_family ≠ "none"` AND `volatility_family ≠ "none"` is present in training
- No C-Holdout sample ID appears in A-Train or B-Train manifests
- No C-Holdout sample is accessible to the model during training (no data loader path, no cache, no preprocessed file)

### 2.2 What This Excludes from Training
- ARMA-GARCH(p,q,r,s) models for any p,q,r,s ≥ 1
- AR-GARCH models (AR mean + GARCH volatility)
- ARMA-ARCH models (ARMA mean + ARCH volatility with s=0)
- Any model with simultaneously non-empty mean and volatility parameter arrays

### 2.3 What Is Permitted in Training (Clarification of Boundary Cases)
- ARMA(p,q) with q=0 is AR — permitted in A-Train
- ARMA(p,q) with p=0 is MA — treatment: TO_BE_DEFINED (must be committed before data generation)
- GARCH(r,s) with s=0 is ARCH — permitted in B-Train
- Zero-mean models with GARCH volatility: mean_family="none", volatility_family="GARCH" — permitted in B-Train

---

## 3. Few-Shot Rules

### 3.1 Few-Shot C at 1%

**Rule:** The training dataset may contain a fraction of C samples equal to 1% of the total training
set size, drawn from a pool that is disjoint from the C-Holdout evaluation set.

**Requirements:**
- The 1% C pool and the C-Holdout are generated from disjoint seeds or disjoint sample ID ranges
  (specific mechanism: TO_BE_DEFINED)
- Explicit seed list for the 1% C pool: TO_BE_DEFINED
- Explicit sample IDs for every C sample in training: listed in `few_shot_1pct_manifest.json`
- Manifest hash committed before training begins
- `C_count_in_few_shot_train` must equal exactly `round(0.01 × total_train_size)` — no rounding ambiguity; exact rule TO_BE_DEFINED
- A-Holdout and B-Holdout sets (for measuring A and B reconstruction quality) must also be specified

### 3.2 Few-Shot C at 5%

**Rule:** Same as §3.1 but with 5% C in training.

**Requirements:**
- The 5% C samples must be drawn from the same disjoint C pool as the 1% samples, but include more samples
- The 5% set must be a strict superset of the 1% set (same samples, plus additional), or a completely disjoint draw — the choice is TO_BE_DEFINED but must be committed before training
- Explicit seed list and sample ID manifest required: `few_shot_5pct_manifest.json`
- Manifest hash committed before training begins

---

## 4. Supervised Upper-Bound Rule

**Rule:** The supervised upper-bound experiment trains on A ∪ B ∪ C-full-train, where C-full-train
is the full C sample pool excluding only the C-Holdout evaluation set.

**Requirements:**
- C-Holdout is strictly withheld in all protocols including supervised
- `C_count_in_supervised_train` is the remaining C pool after C-Holdout is removed
- Explicit sample ID manifest required: `supervised_c_manifest.json`
- This experiment is for reference only — it establishes the ceiling, not a claim about zero-shot generalization

---

## 5. Leakage Guard Requirements

The following verification mechanisms must be implemented before any experiment begins.
Implementation is in a future task (Phase 2 P3); this section documents the requirements.

### 5.1 Family Manifest
Every dataset must be accompanied by a family manifest that lists:
- Total sample count
- Count of A-family samples (split by AR vs ARMA subfamilies)
- Count of B-family samples
- Count of C-family samples
- Any samples with `family_id = "UNKNOWN"` must be rejected (count must be zero)

### 5.2 Sample IDs
Every generated sample must have a unique, deterministic sample ID.
Sample ID generation method: TO_BE_DEFINED (e.g. `SHA256(seed || index)` or UUID from seeded RNG).

### 5.3 Split Manifest Hash
Before training begins, the complete split manifest must be committed to the artifact directory and
its SHA-256 hash must be recorded in the run manifest. Any run whose split manifest hash does not
match the precommitted hash is invalid.

### 5.4 Zero-Shot C Count Assertion
Before any zero-shot training run, the following assertion must pass:

```
assert C_count_in_zero_shot_train == 0, (
    f"SPLIT_CONTRACT_VIOLATION: {C_count_in_zero_shot_train} C samples found in zero-shot training data"
)
```

This assertion must run as the first step of any training script. If it fails, training must stop
immediately and the run must be logged as `SPLIT_VIOLATION`.

### 5.5 Fail-Fast on Split Ambiguity
Any of the following conditions must cause immediate failure before training:
- `family_id = "UNKNOWN"` in any training sample
- A sample ID that appears in both A-Train/B-Train and C-Holdout manifests
- Missing split manifest hash in the run manifest
- Split manifest hash mismatch
- `C_count_in_zero_shot_train > 0`

---

## 6. Correct C-Leakage Framing

The following clarifications prevent the incorrect framing that appeared in the Phase 1 v1 reports:

### 6.1 ARMA Belongs to A-Family, Not C
ARMA models (even those with both p ≥ 1 and q ≥ 1) belong to the A-family because they describe
mean dynamics only, with no volatility component. ARMA in training data is not C-leakage.

### 6.2 C = ARMA-GARCH — Mean AND Volatility Combined
C-leakage means a model with BOTH a non-empty mean component AND a non-empty volatility component
appears in zero-shot training data. This requires:
- `mean_family ∈ {"AR", "ARMA"}` AND `volatility_family = "GARCH"`
- `len(alpha) ≥ 1 AND len(phi) ≥ 1` (at minimum one AR and one ARCH coefficient)

ARMA-only in training data: **not C-leakage**.
GARCH-only in training data: **not C-leakage**.
ARCH-only (GARCH with s=0) in training data: **not C-leakage** (belongs to B-family).
AR-GARCH (AR mean + GARCH volatility) in training data: **IS C-leakage**.

### 6.3 Current State — No Confirmed Leakage, Missing Guard
As documented in `reports/PHASE_2_P1_FAILURE_MAP.md` (v2):
- The canonical Phase 1 generator (`src/data_generator.py`) generates only A and B — no C
- The enriched generator (`src/generate_enriched_training_data.py`) generates ARMA models — belongs to A-family, not C
- **The risk is the absence of a formal split contract and schema-level family guard**, not confirmed leakage
- This document establishes that formal split contract and family guard requirement

---

## 7. Reviewer Approval Required

This split contract must be reviewed and marked APPROVED before any of the following:
- Phase 2 dataset generation begins
- Phase 2 split manifest is committed
- Phase 2 training begins

All TO_BE_DEFINED items must be resolved and documented before approval.

Mark as approved by appending:

```
## APPROVAL RECORD
Approved by: [reviewer name or GitHub handle]
Date: [YYYY-MM-DD]
Commit hash at approval: [full SHA]
Status: APPROVED
Resolved TO_BE_DEFINED items:
  - MA treatment (pure MA in A-Train): [decision]
  - A-Train sample count: [value]
  - B-Train sample count: [value]
  - C-Holdout sample count: [value]
  - Seed for A-Train: [value]
  - Seed for B-Train: [value]
  - Seed for C-Holdout: [value]
  - 1% few-shot C pool mechanism: [decision]
  - 5% few-shot C pool superset rule: [decision]
  - Sample ID generation method: [method]
  - Exact 1% rounding rule: [rule]
```
