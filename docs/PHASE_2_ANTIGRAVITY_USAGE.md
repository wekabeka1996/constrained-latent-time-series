# docs/PHASE_2_ANTIGRAVITY_USAGE.md — Antigravity Tool Usage Guidelines

This document provides operational guidelines for using Antigravity AI agent tool groups in this repository.

## 1. Editor View Use
- **Inspection:** Use the editor/viewer tools primarily for local file inspection, design analysis, and code/markdown editing.
- **Narrow Scope:** Keep modification ranges as narrow as possible. Do not make unrelated changes outside the specified range.
- **Forbidden Drift:** Do not manually drift outside the "Allowed files" list provided in the task prompt. If edits are needed in a forbidden file, stop and ask the user for permission.

## 2. Manager View Use
- **Parallel Tasks:** The manager view (parallel agent coordination) may be used for running parallel independent discovery tasks.
- **Conflict Prevention:** Never run parallel agents on overlapping file sets or branches.
- **Branch Ownership:** Each agent must have a distinct branch, or explicitly coordinated file ownership defined before execution starts.

## 3. Artifacts
Every Antigravity-based code task must produce the following structured artifacts:
- **`implementation_plan.md`**: Created before any source code changes, summarizing design and gates.
- **`task.md`**: Running TODO list to track development progress.
- **`walkthrough.md`**: Created at completion, summarizing modifications and tests.
- **Visual Media:** Capture screenshots or browser recordings *only* when the task requires UI validation (not required for synthetic benchmark tasks).

## 4. Terminal Use
- **Read-Only First:** Prefer read-only commands (e.g., `git status`, `git log`, checking files) before executing modifying actions.
- **No Destructive Commands:** Never execute broad destructive commands (e.g., `rm -rf`, `del /s`, `format`).
- **No Secret Access:** Do not use terminal commands to read `.env` files, config files, keys, or bypass gitignore restrictions.
- **Safe Execution:** Run commands with appropriate limits (e.g., `git log -n 5`) to prevent output buffer flooding.

## 5. Browser Use
- **Sanctioned Lookup:** Browser lookup tools are permitted *only* when the prompt requires external documentation lookup or public reference verification.
- **Source Citations:** If external documentation influences the implementation, summarize the finding and cite the source name in the final report.
- **No Data Uploads:** Using browser sessions for hidden file/data uploads or external exfiltration is strictly prohibited.

## 6. Approval and Human-in-the-Loop
- **Stop on Ambiguity:** If requirements or prompt directions are ambiguous, stop execution and output a clarifying question.
- **Stop on Destructive Actions:** If a task suggests broad file removal, database reformatting, or environment modification, stop and wait for explicit confirmation.
- **Stop on Network Operations:** Do not initiate network calls or downloads except standard git commands required for branch delivery.
- **Stop Before Training:** Never start training runs or execute baseline benchmark runs without explicit user approval.

## 7. Evidence
- **Verification Requirement:** The final report must cite concrete command outputs and commit hashes as evidence of completion.
- **No Narrative-Only Claims:** Narrative statements of task completion must be backed by matching repository changes, branch status prints, and clean working tree checks.
