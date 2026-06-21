# AGENTS.md — Root Governance Rules for Coding Agents

## 1. Repository Mission
This repository is a research reproducibility and Phase 2 constrained latent time-series benchmark project. It is not a production trading system.

## 2. Highest-Priority Rules
- Follow Phase 2 protocol documents strictly before starting any implementation.
- No hidden defaults for research-affecting values.
- No result chasing: all thresholds must be frozen pre-training.
- No training before gates: all gates (G0–G8) must be verified and cleared.
- No model or result claims without explicit, verifiable evidence.
- No C-generation generalization claim is authorized unless all precommitted thresholds in the metric contract pass.
- Negative results are valid scientific outcomes and must be preserved.

## 3. Current Phase State
- Phase 1 (Inventory/Failure Mapping), Phase 2 (Protocol Design), and Phase 3 (Approved Constants) are completed and frozen.
- Phase 4 (Governance Layer) is active.
- Implementation remains BLOCKED.
- Creation of `src/vector_schema_v2.py`, `src/data_generator_v2.py`, `src/vae_constrained.py`, training scripts, yaml configs, datasets, and tests is FORBIDDEN unless explicitly authorized by the user.

## 4. Allowed and Forbidden Actions

| Action | Status | Conditions / Notes |
| :--- | :--- | :--- |
| Documentation edits / report creation | **ALLOWED** | When authorized by task prompt; must only modify/create allowed files. |
| Git commands (`status`, `log`, `branch`, `push`) | **ALLOWED** | Required for branch setup, state check, and branch delivery. |
| Lightweight read-only inspection | **ALLOWED** | For codebase discovery and research. |
| Source code or configuration file creation | **FORBIDDEN** | Unless explicitly authorized by the prompt. |
| Dataset generation / Model training | **FORBIDDEN** | Unless explicitly authorized and after all gates are cleared. |
| Experiment execution / Checkpoint creation | **FORBIDDEN** | Unless explicitly authorized by the task. |
| Artifact overwriting / deletion | **FORBIDDEN** | Run artifacts must never be replaced or removed. |
| Destructive shell commands | **FORBIDDEN** | Never run deletion, formatting, or broad modification commands. |
| Secret/Environment file access | **FORBIDDEN** | Do not read `.env`, credentials, or keys. |
| Unsanctioned network calls | **FORBIDDEN** | Only remote git operations required for delivery are permitted. |

## 5. Config Discipline
- No Pydantic defaults, argparse defaults, dataclass defaults, or function defaults for research-affecting values.
- Every research-affecting value must be explicitly set in the config file snapshot at runtime.
- Missing configuration keys must raise a KeyError immediately and fail fast.
- Legacy baseline values are not Phase 2 defaults. Approved protocol values are defined in `docs/` contract files.

## 6. Git and Evidence Discipline
- Create a dedicated task branch for every prompt task.
- Record the exact base commit hash before making changes.
- Stage and commit only allowed files. The working tree must be clean post-commit.
- Push the branch to remote origin. The final report must verify that local HEAD matches the remote branch hash.
- Document all run commands and test executions. If tests are skipped, record the exact reason.

## 7. Artifact Discipline
- Run snapshots must be immutable. No overwriting run artifacts once written.
- Every future run must produce a manifest, config snapshot, git commit SHA, dirty tree status, environment freeze (`pip freeze`), and contract hashes (`metric_contract_hash.txt`, etc.).
- Documentation-only tasks do not generate run artifacts.

## 8. Safety and Terminal Discipline
- Do not run destructive commands (e.g., `rm -rf`, `del /s`, `format`).
- Do not read or exfiltrate credentials, token strings, or `.env` files.
- Do not use the terminal to bypass ignored or sensitive file restrictions.
- If a command's outcome is ambiguous or potentially broad, stop and request user review.

## 9. Review Gates
- Stop execution after each task is committed and pushed. Do not proceed to the next phase without explicit reviewer approval.
- The final response for every task must include a clear, authorized verdict.

## 10. Pointers to Detailed Docs
- [Protocol Specification](docs/PHASE_2_PROTOCOL.md)
- [Config Discipline Contract](docs/PHASE_2_CONFIG_DISCIPLINE.md)
- [No-Result-Chasing Rules](docs/PHASE_2_NO_RESULT_CHASING_RULES.md)
- [Artifact Contract Specification](docs/PHASE_2_ARTIFACT_CONTRACT.md)
- [Implementation Gates Mapping](docs/PHASE_2_IMPLEMENTATION_GATES.md)
- [Agent Operational Runbook](docs/PHASE_2_AGENT_RUNBOOK.md)
