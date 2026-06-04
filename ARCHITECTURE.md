# Architecture

This repository implements simulation-based inference (SBI) for predator-prey dynamics using the historical Lynx-Hare time series in `data/LynxHare.txt`.

## Runtime Flow

The main workflow is:

1. Load an experiment config with `predator_prey_sbi.config.load_config`.
2. Load observations with `predator_prey_sbi.data.load_lynx_hare`.
3. Build parameter priors with `predator_prey_sbi.priors.build_structure_aware_prior`.
4. Build a simulator function with `predator_prey_sbi.npe.build_simulator`.
5. Train a neural posterior estimator through `predator_prey_sbi.infer`.
6. Evaluate posterior predictive, residual, latent, and SBC diagnostics through `predator_prey_sbi.diagnostics`.

Experiment YAML files live in `configs/experiments/` and generally extend `configs/base.yaml`.

## Source Layout

- `predator_prey_sbi/config.py` loads and merges YAML configs.
- `predator_prey_sbi/data.py` loads the observed Lynx-Hare data.
- `predator_prey_sbi/simulator.py` implements the predator-prey simulator and observation model.
- `predator_prey_sbi/parameters.py` resolves inferred parameter dictionaries into simulator parameters.
- `predator_prey_sbi/priors.py` builds data-informed prior bounds and parameter order.
- `predator_prey_sbi/features.py` builds summary or embedding inputs for neural SBI.
- `predator_prey_sbi/npe.py` wraps `sbi` training and simulator construction.
- `predator_prey_sbi/infer.py` is the inference CLI.
- `predator_prey_sbi/diagnostics.py` is the diagnostics CLI.

## Documentation Layout

- `AGENTS.md` contains repository instructions for coding agents.
- `ARCHITECTURE.md` is this high-level technical map.
- `docs/PLANS.md` defines the ExecPlan format.
- `docs/exec-plans/active/` contains active execution plans.
- `docs/exec-plans/completed/` contains completed execution plans.
- `docs/exec-plans/lab-journal.md` summarizes experiment outcomes.
- `docs/exec-plans/lab-journal.tsv` stores machine-readable experiment result rows.
- `docs/exec-plans/scripts/` contains helper scripts for batch experiments.
