# Exec Plans

This directory contains the planning and experiment record for the predator-prey SBI work.

## Layout

- `active/` contains ExecPlans that still describe ongoing work.
- `completed/` contains ExecPlans whose implementation and first validation pass are complete.
- `lab-journal.md` is the narrative experiment journal.
- `lab-journal.tsv` is the machine-readable experiment result table used for RMSE comparisons.
- `scripts/` contains helper scripts for running experiment batches and appending rows to `lab-journal.tsv`.

New complex features should start as an ExecPlan in `active/`. When the plan is completed and its outcome is recorded, move it to `completed/` with `git mv`.
