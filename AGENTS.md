# AGENTS.md

Production ready scientific code for implementing Simulation-Based Inference (SBI) for Predatory-Prey Dynamics.

## Code style

Strict python. Use ty for type checking and ruff for formatting and linting.

Always run formatting and linting with ruff, and type checking with ty.

Always use conventional commits for commit messages.

The agent working on this repository can freely choose which python packages to install.

Dependency management is handled entirely by uv; do not use pip.

# ExecPlans
 
When writing complex features or significant refactors, use an ExecPlan (as described in docs/PLANS.md) from design to implementation.

Store active plans in docs/exec-plans/active/ and completed plans in docs/exec-plans/completed/. Record experiment outcomes in docs/exec-plans/lab-journal.md and docs/exec-plans/lab-journal.tsv.
