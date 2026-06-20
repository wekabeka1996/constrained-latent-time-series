# PHASE 2 SCHEMA SPECIFICATION
# Vector Schema v2 Intent — Documentation Only, No Code

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval before any implementation begins

> [!IMPORTANT]
> This document is a **documentation-only schema intent specification**.
> No code is created or modified by this document.
> `src/vector_schema_v2.py` does not exist and must not be created until this document
> is reviewed and marked APPROVED.

---

## 1. Why the Legacy 40D Schema Is Invalid for C

The existing Phase 1 parameter vector is a 40-dimensional array with the following layout
(from `src/vector_schema.py`):

```
Indices  0– 9 : structural indicators (continuous logits for family membership)
Indices 10–29 : parameter block (family-dependent interpretation)
Indices 30–39 : statistical moment summary
```

This schema has **three structural defects** that prevent it from being used for C-family
(ARMA-GARCH) representation:

### Defect 1: v[10] Collision

```
For ARMA models:   v[10] = φ₁   (first AR coefficient)
For GARCH models:  v[10] = ω    (GARCH intercept / unconditional variance)
```

The same memory slot carries a different mathematical meaning depending on which family is active.
This is a context-dependent overload. Under the legacy schema, an ARMA-GARCH vector would need both
φ₁ and ω simultaneously, but they share the same index — making a joint representation
**formally undefined**.

### Defect 2: Family-Dependent Interpretation of Same Indices

The parameter block (indices 10–29) is interpreted as AR coefficients for ARMA models and as
ω/α/β arrays for GARCH models. There is no static layout that is valid for both families simultaneously.

### Defect 3: Undefined ARMA-GARCH Layout

As a direct consequence of defects 1 and 2, the schema contains no valid encoding for an
ARMA-GARCH vector. Any decoded latent point with both ARMA and GARCH indicators weakly active
produces a vector in `UNKNOWN` state (not a valid C sample, not a valid A or B sample).

This was confirmed in Phase 1: 0/200 reconstruction validity (all `LEGACY_LAYOUT_COLLISION`).

---

## 2. Requirements for Schema v2

A Phase 2 schema must satisfy all of the following requirements. These are requirements only —
they do not specify implementation choices.

### R1 — Explicit Family Identifier
Every parameter vector must carry an unambiguous family identifier that determines which
component slots are valid. The identifier must be deterministic from the schema, not inferred
from thresholds.

### R2 — Separate Mean Component
The mean dynamics component (ARMA parameters: orders p, q; coefficients φ₁…φₚ, θ₁…θ_q)
must occupy a dedicated, non-overlapping slot range in the vector layout.

### R3 — Separate Volatility Component
The volatility dynamics component (GARCH parameters: orders r, s; coefficients ω, α₁…αᵣ, β₁…βₛ)
must occupy a dedicated, non-overlapping slot range in the vector layout.

### R4 — No Overlapping Parameter Slots
No index in the vector may be used by both the mean component and the volatility component.
The A-region and B-region are disjoint by construction.

### R5 — Explicit Orders p/q/r/s
The lag orders p (AR), q (MA), r (ARCH), s (GARCH) must be explicit fields, not hidden inside
normalized continuous logit representations unless the mapping is formally specified and invertible.

### R6 — Explicit Parameter Arrays
Each coefficient array (φ, θ, α, β) must have a known offset and a known maximum length.
Elements beyond the order must be zero-padded, not left undefined.

### R7 — Explicit Validity Constraints
The schema specification must list the mathematical constraints that any vector must satisfy
to be a valid sample for each family:
- A: AR stationarity, MA invertibility
- B: GARCH positivity (ω > 0, αᵢ ≥ 0, βⱼ ≥ 0), GARCH persistence (Σα + Σβ < 1)
- C: simultaneously A and B valid

### R8 — Explicit Serialization Rules
The schema must specify how a ModelSpec logical object (see §3) maps to a fixed-size numeric vector
and back. This mapping must be bijective for valid samples.

### R9 — No Raw Integer Orders Hidden in Continuous Logits
The legacy schema mixes integer lag orders with continuous logit representations in ways that are
not invertible without additional context. The Phase 2 schema must either:
- Use integer orders stored at fixed positions with specified range, OR
- Use one-hot/multi-hot order encoding at specified positions, OR
- Use a documented soft-order representation with a specified decoding function
Any of these is acceptable, but the choice must be explicit and specified before implementation.

---

## 3. Required Logical Object: ModelSpec

The following is a **logical object definition** — not a code class, not a Pydantic model.
It describes the conceptual structure that the Phase 2 schema must be able to represent.

Implementation form (Python dataclass, Pydantic, TypedDict, class, etc.) is BLOCKING_TO_BE_DEFINED and remains deferred in P3 as ModelSpec is a logical protocol object only.

```
ModelSpec
├── family_id:           str        — one of {"AR", "ARMA", "GARCH", "ARMA-GARCH"}
├── mean_family:         str        — one of {"none", "AR", "ARMA"}; "none" for pure GARCH
├── volatility_family:   str        — one of {"none", "GARCH"}; "none" for pure ARMA
├── mean_order:
│   ├── p:               int ≥ 0   — AR order; 0 if no mean component
│   └── q:               int ≥ 0   — MA order; 0 if no mean component
├── volatility_order:
│   ├── r:               int ≥ 0   — ARCH order; 0 if no volatility component
│   └── s:               int ≥ 0   — GARCH order; 0 if no volatility component
├── mean_params:
│   ├── phi:             float[p]  — AR coefficients; empty array if p=0
│   └── theta:           float[q]  — MA coefficients; empty array if q=0
├── volatility_params:
│   ├── omega:           float     — GARCH intercept; 0.0 if no volatility component
│   ├── alpha:           float[r]  — ARCH coefficients; empty array if r=0
│   └── beta:            float[s]  — GARCH coefficients; empty array if s=0
├── constraints:
│   ├── ar_stationary:          bool  — φ polynomial roots inside unit circle
│   ├── ma_invertible:          bool  — θ polynomial roots outside unit circle
│   ├── garch_positive:         bool  — ω>0, all αᵢ≥0, all βⱼ≥0
│   ├── garch_persistence_ok:   bool  — Σα + Σβ < 1
│   └── is_valid_C:             bool  — ar_stationary AND ma_invertible AND garch_positive AND garch_persistence_ok
└── provenance:
    ├── generator_mode:   str    — one of {"synthetic_arma", "synthetic_garch", "synthetic_arma_garch"}
    ├── seed:             int    — explicit seed used for this sample
    └── sample_id:        str    — unique ID for this sample (for split manifests)
```

### Constraints on ModelSpec

- `family_id = "AR"`:   mean_family="AR", volatility_family="none", q=0, r=0, s=0
- `family_id = "ARMA"`: mean_family="ARMA", volatility_family="none", r=0, s=0
- `family_id = "GARCH"`: mean_family="none", volatility_family="GARCH", p=0, q=0
- `family_id = "ARMA-GARCH"`: mean_family ∈ {"AR","ARMA"}, volatility_family="GARCH", all orders ≥ 1

Violation of any constraint must raise a validation error, not silently produce a default.

---

## 4. Required Represented Families

The Phase 2 schema must be able to validly represent all four families:

### AR (subfamily of A)
- `mean_family = "AR"`, `volatility_family = "none"`
- φ polynomial: all roots strictly inside unit circle
- MA coefficients: empty (q=0)
- GARCH parameters: all zero/empty

### ARMA (A-family)
- `mean_family = "ARMA"`, `volatility_family = "none"`
- φ polynomial: all roots strictly inside unit circle (stationarity)
- θ polynomial: all roots strictly outside unit circle (invertibility)
- GARCH parameters: all zero/empty

### GARCH (B-family)
- `mean_family = "none"`, `volatility_family = "GARCH"`
- AR/MA coefficients: all zero/empty
- ω > 0, all αᵢ ≥ 0, all βⱼ ≥ 0, Σα + Σβ < 1

### ARMA-GARCH (C-family)
- `mean_family ∈ {"AR", "ARMA"}`, `volatility_family = "GARCH"`
- Simultaneously satisfies ARMA validity AND GARCH validity
- All constraints in `ModelSpec.constraints` must be True
- This family is completely absent from zero-shot training data

---

## 5. Forbidden Schema Patterns

The following patterns are explicitly forbidden in Phase 2 schema design:

### FP1 — Overloaded Numeric Slots
Using the same vector index to represent different parameters depending on context.
**Example of violation:** v[10] = φ₁ for ARMA, v[10] = ω for GARCH.

### FP2 — Family-Dependent Interpretation of Same Index
Interpreting the same index as carrying different information for different family types.
Any index with a family-conditional meaning is forbidden.

### FP3 — Implicit Family from Threshold Only
Deriving family membership by comparing a continuous logit against a threshold (e.g. 0.5),
where the threshold is a function parameter default and not verifiably consistent.
Family membership must be explicit in the vector structure, not inferred from continuous values
via an unspecified threshold.

### FP4 — Hidden Default Order
Any schema that uses a default maximum lag order (e.g. max_lag_order=5) without making the
actual order of each sample explicit in the vector. The order of each sample must be deterministic
from the vector itself, not from a globally assumed maximum.

### FP5 — Hidden Default max_lag
Same as FP4 — the maximum lag bound must be an explicit schema constant, not a module-level
default or a Pydantic default.

### FP6 — Mixed Label Types
Using integer labels for some samples (e.g. `2` for ARMA as in `generate_enriched_training_data.py`)
and string labels for others (e.g. `"AR"`, `"GARCH"`) in the same data pipeline. All labels
must be the same type and drawn from the same vocabulary.

---

## 6. Phase 2 Schema Dimension Specification

The exact dimensions of the Phase 2 vector are specified as follows:
- APPROVED_PROTOCOL_VALUE: max_p = 5
- APPROVED_PROTOCOL_VALUE: max_q = 5
- APPROVED_PROTOCOL_VALUE: max_r = 2
- APPROVED_PROTOCOL_VALUE: max_s = 2
- APPROVED_PROTOCOL_VALUE: schema_v2_min_flat_dim = 32

### Rationale
- p/q up to 5 is sufficient for ARMA mean dynamics and matches the legacy max lag scale.
- r/s up to 2 covers GARCH(1,1), GARCH(2,1), GARCH(1,2), and GARCH(2,2) without making the benchmark too large.
- Keeps Phase 2 feasible on 8GB VRAM and CPU-based synthetic generation.
- schema_v2_min_flat_dim = 32 is a logical boundary serialization minimum, not the internal representation.

### ModelSpec Logical Object Fields
The logical object must support the following fields:
- family_id
- mean_family
- volatility_family
- p
- q
- r
- s
- ar_params[max_p]
- ma_params[max_q]
- omega
- alpha_params[max_r]
- beta_params[max_s]
- constraint_flags
- provenance fields

### Logical Serialization Layout Intent
This layout represents a **logical serialization intent, not implementation code**.

- index 0: family_id enum/logit source, not final truth
- index 1: mean_family enum/logit source
- index 2: volatility_family enum/logit source
- index 3: p
- index 4: q
- index 5: r
- index 6: s
- index 7: omega
- indices 8:12: reserved constraint/provenance flags
- indices 12:17: ar_params[5]
- indices 17:22: ma_params[5]
- indices 22:24: alpha_params[2]
- indices 24:26: beta_params[2]
- indices 26:32: reserved explicit diagnostics/provenance slots

> [!WARNING]
> The final implementation may use a typed ModelSpec object internally and only serialize to flat vectors at model boundaries. The protocol forbids overloaded numeric slots and family-dependent interpretation of the same index.

---

## 7. Relationship to Existing Code

### `src/vector_schema.py`
This file defines the Phase 1 legacy 40D schema. It must NOT be modified for Phase 2.
Phase 2 must use a new schema (to be implemented in a future task as `src/vector_schema_v2.py`
or equivalent). The Phase 1 schema remains valid for Phase 1 baselines.

### `src/validation.py`
The validity functions (`is_stationary_ar`, `is_invertible_ma`, `is_valid_garch`) in this file
implement the mathematical constraints listed in §2 R7. They are Phase 1 functions and may be
reused for Phase 2 validation **if and only if** their function-level defaults (tol=1e-8,
threshold=0.5, epsilon=0.05) are replaced with config-driven values in Phase 2 callers.

### `src/data_generator.py`
This generator produces A and B families only. It must NOT be used to generate C-family samples
for Phase 2 training or evaluation. A new generator (to be implemented in a future task) must
produce all four families under the Phase 2 schema.

---

## 8. Reviewer Approval Required

This schema specification document must be reviewed and marked APPROVED before:
- `src/vector_schema_v2.py` is created
- Any Phase 2 data generator is created
- Any Phase 2 model is created
- Any Phase 2 training run is started

Mark as approved by appending the following block to this document:

```
## APPROVAL RECORD
Approved by: [reviewer name or GitHub handle]
Date: [YYYY-MM-DD]
Commit hash at approval: [full SHA]
Status: APPROVED
```
