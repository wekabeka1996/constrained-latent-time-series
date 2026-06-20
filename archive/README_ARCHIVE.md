# Archive

This folder contains legacy exploratory code, invalidated artifacts, and
historical outputs retained for audit history only.

These files are NOT part of the clean reproduction pipeline defined in
`src/reproduce.py`.

## Why these files are retained

- To preserve audit history for the claim-by-claim verdict in
  `docs/CLAIM_VERDICT_MATRIX.md`.
- To provide provenance for the invalidated Fisher metric outputs.
- To retain the original research article for citation purposes.
- To keep experimental code reachable for future reference.

## What is NOT safe to cite from this archive

- `invalidated_geometry/` — Fisher metric outputs are mathematically
  invalid due to a Jacobian bug. See `docs/GEOMETRY_VALIDITY_STATUS.md`.
- `original_research/Стаття_Результат_Дослідження.md` — The article
  contains claims that were partially or fully invalidated by the clean
  reproduction. See `docs/CLAIM_VERDICT_MATRIX.md`.
- `legacy_results/generation_1/fisher_metrics/` — Same as above.
- `orphan_artifacts/generated_valid_thetas.npy` — A forbidden artifact
  from the legacy generator. Do not use as evidence.

## Integrity policy

- **No Active Pipeline**: The archive is NOT part of the active reproduction pipeline.
- **Do Not Cite Invalidated Geometry**: Any geometry metrics in this archive are mathematically invalid due to a Jacobian bug.
- **Original Article Context**: The original research article contains claims that were partially or fully invalidated by the clean reproduction.
- **Orphan Artifacts**: All archived artifacts are historical only and should not be used as active evidence.
- **No Lint/Test Guarantees**: Legacy code in the archive is not active and may not pass linting or pytest suites.
- **Local-Only Binary Policy (Policy B)**: To prevent repository bloat, large binary legacy artifacts (`.npy`, `.csv`, `.png`, `.pth`) in `archive/` are local-only and excluded from public git tracking via the root `.gitignore`. Only code and markdown files in the archive are committed to public source control.
