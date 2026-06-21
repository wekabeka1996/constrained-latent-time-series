# docs/PHASE_2_REVIEW_CHECKLIST.md — Agent Work Review Checklist

This checklist is used by reviewers (and must be self-checked by agents before submission) to verify task completion and maintain research governance.

## 1. Documentation Task Checklist
- [ ] Only files in `docs/` and `reports/` directories were modified or created.
- [ ] No Python source code (`src/` or elsewhere) was modified or created.
- [ ] No test files (`tests/`) were modified or created.
- [ ] No configuration snapshots (`configs/`) were modified or created.
- [ ] No database files or mathematical datasets were generated.
- [ ] No model training pipelines or weights were executed or changed.
- [ ] Git branch evidence is present (HEAD hash, status, log).
- [ ] Remote branch hash matches the local HEAD hash exactly.
- [ ] The final report contains a clear and justified verdict choice.

## 2. Implementation Task Checklist
- [ ] The task scope strictly matches the limits set in the prompt.
- [ ] The root instructions in `AGENTS.md` were followed.
- [ ] Relevant protocol contract documents are referenced by name.
- [ ] No hidden defaults exist in newly implemented functions or classes.
- [ ] New unit tests are added to verify correctness of the implemented feature.
- [ ] All unit tests pass, and skips are justified.
- [ ] Required snapshotted artifacts (manifest, environment log) are generated.
- [ ] No spontaneous generalization or success claims are made.

## 3. Config Review Checklist
- [ ] Every configuration parameter is explicitly declared; no defaults are fallback-selected.
- [ ] Command line parser does not declare defaults for research parameters.
- [ ] Pydantic/dataclass schema definitions use required field declarations (e.g. `...` or no default) for research values.
- [ ] Function parameters for validation check thresholds receive explicit config inputs.
- [ ] Config loader fails fast and raises a KeyError on any missing parameter key.

## 4. Research Integrity Checklist
- [ ] No post-hoc modification of metric code or threshold values occurred.
- [ ] All success/failure conditions are precommitted and documented.
- [ ] Negative results or failed runs are documented and preserved.
- [ ] The C-family leakage guard assertion `assert C_count_in_zero_shot_train == 0` is present.

## 5. Safety Checklist
- [ ] No broad or destructive shell commands (e.g. `rm -rf`, `del /s`, `format`) were executed.
- [ ] No credential paths, `.env` files, or token files were read or modified.
- [ ] No unsanctioned external network uploads or API requests occurred.
- [ ] No attempts were made to bypass repository `.gitignore` or access permissions.
- [ ] Broad deletions or modifications were stopped for human-in-the-loop review.

## 6. Git Checklist
- [ ] Base commit hash is verified and matched before making edits.
- [ ] Branch naming convention matches `phase2/p<N>-<name>`.
- [ ] The final report lists exactly which files changed and verifies they match the prompt bounds.
- [ ] The working tree is clean (no untracked files or uncommitted changes) after commit.
- [ ] Remote push check confirms that the local commit is successfully synchronized to origin.

## 7. Review Verdict Options

The reviewer must choose one of the following verdicts for each task submission:

| Verdict | Meaning / Action Required |
| :--- | :--- |
| **`ACCEPTED`** | Task is fully complete, verified, and branch is ready to merge. |
| **`ACCEPTED_WITH_NOTES`** | Task is acceptable, minor doc polish needed, but does not block progress. |
| **`REJECTED_FIX_REQUIRED`** | Task has errors or missing evidence. The agent must checkout the same branch, fix errors, amend commit, force-push, and resubmit. |
| **`BLOCKED_BY_SCOPE_VIOLATION`** | The agent modified forbidden files or created unauthorized code/configs. Revert and redo. |
| **`BLOCKED_BY_MISSING_EVIDENCE`** | Git hashes, log evidence, or status prints are missing or mismatch. |
| **`BLOCKED_BY_UNSAFE_ACTION`** | The agent attempted destructive terminal commands or credential reading. RED ALARM. |
