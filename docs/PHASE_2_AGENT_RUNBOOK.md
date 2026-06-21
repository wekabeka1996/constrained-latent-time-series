# docs/PHASE_2_AGENT_RUNBOOK.md — Agent Operational Runbook

## 1. Purpose
This runbook provides detailed operational guidelines for future Antigravity coding agents working in this repository. While `AGENTS.md` acts as the root rule file containing high-level imperative constraints, this runbook outlines the day-to-day execution lifecycle, workflow assumptions, templates, and safety limits.

## 2. Antigravity Workflow Assumptions
- **Contexts:** Agents operate within editor (file mutation), terminal (git operations, read-only checks, testing), and browser (documentation lookup) contexts.
- **Output Artifacts:** Tasks may generate plan files (`implementation_plan.md`), task lists (`task.md`), code diffs, command execution records, and completion reports.
- **Verification Rule:** Narrative claims of success are insufficient. All claims must be backed by verifiable outputs (git commit hashes, status printouts, file structures, test results).
- **Parallel Work Coordination:** Multiple agents operating simultaneously must work on separate branches and disjoint files. Simultaneous modification of overlapping file sets is forbidden unless explicitly coordinated.

## 3. Task Lifecycle
Coding agents must execute tasks using the following 10-step lifecycle:
1. **Start-State Verification:** Fetch origin and verify HEAD commit aligns with the base target.
2. **Branch Creation:** Check out a new branch matching naming conventions.
3. **Scope Check:** Inspect prompt boundaries to identify allowed vs. forbidden actions.
4. **Allowed Files Check:** Ensure that only files on the "Allowed files" list will be created or modified.
5. **Work Execution:** Make modifications, ensuring compliance with protocol contracts and coding standards.
6. **Evidence Capture:** Record commands executed, file changes, and status.
7. **Testing / No-Test Rationale:** Execute relevant test suites. If skipped, document the exact environment/library blockers.
8. **Commit:** Stage and commit only allowed files. Ensure a clean working tree post-commit.
9. **Push:** Push the branch to the remote origin.
10. **Post-Push Verification:** Run commands to verify that local HEAD matches the remote branch hash, then pause for human review.

## 4. Required Start-State Verification Template
Before starting any work, the agent must execute the following verification commands:
```bash
git fetch origin
git checkout <base_branch>
git pull --ff-only
git log --oneline -5
git status --short
```
Verify that the output of `git log` contains the expected base commit hash. If it does not, stop work immediately and report the block to the user.

## 5. Branch Naming Convention
Task branches must follow this naming convention:
`phase2/p<phase_number>-<descriptive-short-name>`

Examples:
- `phase2/p4-agent-governance`
- `phase2/p5-schema-guard-skeleton`
- `phase2/p6-generator-v2`
- `phase2/p7-dataset-split-guards`

## 6. Commit Message Convention
Commit messages must follow this convention:
`PHASE2_P<phase_number> <descriptive action message>`

Examples:
- `PHASE2_P4 agent governance and runbook`
- `PHASE2_P5 schema guard skeleton`
- `PHASE2_P6 generator v2 and constraints`

## 7. Final Report Template
At the end of every task, the agent's final response must include a report with the following structure:
```markdown
### TASK COMPLETION REPORT: [Task Name]

- **Branch Name:** [e.g. phase2/p4-agent-governance]
- **Base Commit Verified:** [e.g. 04dc202a633009f9fd04483f5589818991799187]
- **Commit Hash:** [local commit SHA]
- **Remote Branch Hash:** [pushed remote SHA]
- **Changed Files:**
  - [file 1]
  - [file 2]
- **Files Not Changed:**
  - [directory/file status confirmation]
- **Commands Run:**
  - [command 1]
  - [command 2]
- **Tests Run or Exact Reason Not Run:**
  - [test execution status and detail]
- **Evidence Collected:**
  - [git logs, status output, environment details]
- **Known Limitations:**
  - [any warnings or missing dependencies]
- **Remaining Blockers:**
  - [list of blockers]
- **Final Verdict:** [READY_FOR_REVIEW | BLOCKED]
```

## 8. Antigravity Artifact Expectations
When executing code tasks, future agents must generate and update the following artifacts:
- **`implementation_plan.md`**: Design plan created before making code changes, requiring user review/feedback.
- **`task.md`**: Living TODO list tracking checklist items (`[ ]`, `[/]`, `[x]`).
- **`walkthrough.md`**: Created after completion, summarizing modifications, test evidence, and validation.

## 9. Parallel Agent Rules
- Only one agent may work on a branch at any time.
- No two agents may edit the same set of files concurrently.
- Cross-branch cherry-picking is prohibited unless explicitly directed by the user.
- The final report must explicitly declare whether the task was completed by a single agent or in coordination with parallel agents.

## 10. Human Review Gates
- Agents must stop execution after pushing a branch.
- Automatic transition or chaining of prompts without user review is prohibited.
- If the reviewer rejects a task, the agent must execute only the required fixes under the same or a specific correction branch.
