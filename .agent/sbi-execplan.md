# Simulation-Based Inference for Lynx-Hare Predator-Prey Dynamics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository includes .agent/PLANS.md from the repository root. This ExecPlan must be maintained in accordance with .agent/PLANS.md.

## Purpose / Big Picture

The goal is to let a user infer predator-prey model parameters from the historical lynx-hare time series in data/LynxHare.txt using neural simulation-based inference (SBI). After this change, a user can run a single command to train a neural posterior estimator, generate a posterior distribution over model parameters, and finally run posterior predictive simulations to see whether the inferred parameters reproduce the observed cycles. Success is observable by producing posterior samples, plots of simulated trajectories that overlap the observed data, and diagnostics that demonstrate the inference is not obviously biased.

## Progress

- [x] (2026-01-02 20:57Z) Read repository context, data file, and PLANS.md; drafted initial ExecPlan.
- [x] (2026-01-02 21:01Z) Committed to neural SBI (sbi + torch) as the inference approach.
- [x] (2026-01-04 17:35Z) Added torch/sbi dependencies and completed the neural SBI smoke test.
- [x] (2026-01-03 06:31Z) Implemented data loading, Lotka-Volterra simulator, log-noise observation model, and simulation CLI/config.
- [x] (2026-01-04 18:53Z) Implemented neural SBI inference pipeline and ran it on Lynx-Hare data.
- [x] (2026-01-04 19:02Z) Added diagnostics (posterior predictive + SBC) and validated with a full diagnostics run.

## Surprises & Discoveries

- Observation: README.md is empty and main.py is a placeholder; the repository has no existing SBI code to extend.
  Evidence: main.py prints a greeting, and README.md has no content.
- Observation: data/LynxHare.txt appears to be a three-column yearly time series without a header.
  Evidence: first lines show year followed by two numeric columns.
- Observation: Importing sbi triggers arviz, which attempts to create Path.home()/arviz_data and fails in a sandboxed home directory.
  Evidence: PermissionError when importing sbi before setting HOME to a writable path.
- Observation: The inference pipeline completes with the default configuration in ~20 seconds for 500 simulations on this machine.
  Evidence: make infer produced a posterior_samples.npz after training 500 simulations.
- Observation: Diagnostics completed with posterior predictive RMSE of ~49 (hare) and ~22 (lynx) and SBC coverage between 0.65 and 0.85 for default settings.
  Evidence: diagnostics_metrics.json in runs/2026-01-04_190213.

## Decision Log

- Decision: Use the classic Lotka-Volterra predator-prey differential equations as the simulator baseline, with parameters alpha (prey growth), beta (predation rate), delta (predator growth from predation), and gamma (predator death).
  Rationale: This model is the standard baseline for lynx-hare cycles and is simple enough to simulate quickly for SBI.
  Date/Author: 2026-01-02, Codex
- Decision: Use neural SBI (sbi + torch) as the sole inference approach, with a neural posterior estimator trained on summary statistics of simulated trajectories.
  Rationale: The user explicitly requested neural SBI, and summary-statistic inputs keep the neural pipeline simple and stable for a novice to run.
  Date/Author: 2026-01-02, Codex
- Decision: Use hand-crafted summary statistics (for example, mean, variance, autocorrelation, peak counts, and lagged cross-correlation) as the inference inputs rather than a learned embedding network.
  Rationale: The user confirmed a summary-statistics approach for now; this reduces complexity and speeds iteration while still enabling neural posterior estimation.
  Date/Author: 2026-01-03, Codex
- Decision: Set HOME and MPLCONFIGDIR to repo-local writable directories when running SBI tools to avoid arviz and matplotlib cache permission errors.
  Rationale: sbi imports arviz, which writes to Path.home(); ensuring a writable HOME prevents failures in the sandboxed environment.
  Date/Author: 2026-01-04, Codex
- Decision: Assume the second column in data/LynxHare.txt is the prey (hare) series and the third column is the predator (lynx) series, with units treated as relative counts.
  Rationale: This is the common ordering for the lynx-hare dataset; the plan remains flexible if a different ordering is confirmed.
  Date/Author: 2026-01-02, Codex

## Outcomes & Retrospective

Work has not started yet. The initial ExecPlan defines scope, milestones, and acceptance criteria.

## Context and Orientation

The repository currently contains main.py (a placeholder), pyproject.toml (no dependencies), data/LynxHare.txt (observed time series), README.md (empty), and agents.md. There is no Python package structure or inference code. The plan will add a small Python package to keep simulation, inference, and diagnostics organized.

Terms used in this plan are defined here in plain language:

Simulation-based inference (SBI) means using a simulator to generate synthetic data for guessed parameter values, then using those simulations to learn which parameters are most plausible given the real observations.

A simulator is code that takes model parameters and produces a time series of predator and prey population values.

The prior is the initial range or distribution of parameters we are willing to consider before seeing the data.

The posterior is the updated distribution of parameters after comparing simulations to observations.

An observation model describes how simulated values become observed data, for example by adding noise.

Summary statistics are simplified measurements of a time series (for example, mean, variance, or autocorrelation) used to compare simulations when full trajectories are too noisy or high-dimensional.

## Plan of Work

Milestone 1 focuses on creating a minimal, correct simulator and data pipeline. The repository will gain a package (for example predator_prey_sbi/) containing: a data loader for data/LynxHare.txt, a Lotka-Volterra ODE simulator, and an observation model that adds noise in log space to mimic measurement error. This milestone should end with a command that runs the simulator with fixed parameters and plots or prints the resulting time series so a human can see the cycles.

Milestone 2 establishes the neural SBI stack. It installs torch and sbi, adds a small smoke-test module (for example predator_prey_sbi/smoke.py) that trains a tiny neural posterior estimator on synthetic data, and records any dependency quirks in Surprises & Discoveries. This milestone ends with a working, reproducible command that confirms the stack can run on this environment.

Milestone 3 implements the full inference pipeline. It trains a neural posterior estimator that maps summary statistics of simulated trajectories to a posterior over parameters. The pipeline will include a way to run multiple simulations, compare to the observed data, and save posterior samples to disk.

Milestone 4 adds diagnostics and validation. This includes posterior predictive checks (simulate from the posterior and compare to observed data), simulation-based calibration (SBC) on synthetic datasets, and simple coverage checks (how often the true parameters are within credible intervals for synthetic data). The milestone ends with a reproducible command that runs diagnostics and produces a short report or plot, confirming that the inference is at least qualitatively reasonable.

Milestone 5 cleans up the user workflow: add a CLI entry point (python -m ...) with a configuration file, update README.md with usage, and add tests for core pieces like the simulator and summary-statistics feature extraction. The deliverable is a single end-to-end command that trains/infers and writes posterior samples and plots, and tests that pass with ruff and ty checks.

## Concrete Steps

All commands assume the working directory is /Users/robertsiipola/predator-prey-sbi.

1. Create the package structure and data loader, then run a small simulation.
   Command examples:
     python -m predator_prey_sbi.simulate --config configs/base.yaml
   Expected output (example):
     Loaded data/LynxHare.txt with 91 yearly observations.
     Simulated 91 steps using Lotka-Volterra parameters: alpha=1.5, beta=0.9, delta=0.75, gamma=1.3
     Wrote plots to runs/2026-01-02/simulated_trajectories.png

2. Run the neural SBI smoke test after dependencies are installed.
   Command examples:
     python -m predator_prey_sbi.smoke
   Expected output (example):
     Neural SBI smoke test completed with 200 simulations.
     Saved test posterior samples to runs/2026-01-02/smoke_posterior_samples.npz

3. Run the inference pipeline for the observed data.
   Command examples:
     python -m predator_prey_sbi.infer --config configs/base.yaml --observed data/LynxHare.txt
   Expected output (example):
     Training neural posterior estimator with sbi on 10,000 simulations
     Saved posterior samples to runs/2026-01-02/posterior_samples.npz

4. Run diagnostics and posterior predictive checks.
   Command examples:
     python -m predator_prey_sbi.diagnostics --config configs/base.yaml --posterior runs/2026-01-02/posterior_samples.npz
   Expected output (example):
     Posterior predictive RMSE (hare): 6.2; (lynx): 5.9
     SBC rank histogram saved to runs/2026-01-02/sbc_rank.png

## Validation and Acceptance

The work is accepted when a user can run the neural SBI smoke test and the full inference command to infer parameters from data/LynxHare.txt, and see posterior samples and posterior predictive plots produced in a runs/ timestamped directory. The simulator must reproduce oscillatory predator-prey behavior for at least one known parameter set, and posterior predictive trajectories generated from the inferred posterior must visibly overlap the observed data. Tests must pass: running ruff (format and lint) and ty (type checks) should succeed, and new unit tests should fail before the change and pass after.

Acceptance includes that neural posterior estimation completes for a small simulation budget (for example 1,000 simulations) and produces posterior samples, and that posterior predictive plots show similar cyclic behavior to the observed data.

## Idempotence and Recovery

All commands should be safe to re-run; they must overwrite or create a new timestamped output directory without corrupting previous results. If installation or inference fails, the recovery path is to delete only the failed run directory under runs/ and retry, then adjust dependency versions in pyproject.toml as described in the plan (for example, pin torch and sbi to a compatible pair) and rerun the smoke test.

## Artifacts and Notes

Expected minimal artifacts after Milestone 3:
  runs/<timestamp>/posterior_samples.npz containing arrays for alpha, beta, delta, gamma.
  runs/<timestamp>/simulated_trajectories.png showing observed versus simulated time series.

A short example of the expected posterior_samples.npz keys (for reference):
  ['alpha', 'beta', 'delta', 'gamma']

## Interfaces and Dependencies

Dependencies will be added to pyproject.toml. The baseline set is NumPy and SciPy; Matplotlib is used for plots. Add torch and sbi for neural posterior estimation.

Define the following interfaces in the new package predator_prey_sbi:

In predator_prey_sbi/data.py, define:
  def load_lynx_hare(path: str) -> tuple[list[float], list[float], list[float]]:
      """Return (years, hare, lynx) arrays from a whitespace-delimited file."""

In predator_prey_sbi/simulator.py, define:
  def simulate_lv(
      years: list[float],
      params: dict[str, float],
      x0: tuple[float, float],
      dt: float,
      noise_scale: float,
      rng_seed: int | None,
  ) -> tuple[list[float], list[float]]:
      """Simulate prey/predator series with Lotka-Volterra ODE and log-normal observation noise."""

In predator_prey_sbi/features.py, define:
  def summarize_series(hare: list[float], lynx: list[float]) -> list[float]:
      """Compute summary statistics used as inputs to the neural posterior estimator."""

In predator_prey_sbi/npe.py, define:
  def train_posterior(
      sims: list[dict[str, float]],
      summaries: list[list[float]],
      rng_seed: int | None,
  ) -> object:
      """Train a neural posterior estimator and return a sampler object."""

In predator_prey_sbi/smoke.py, define:
  def run_smoke_test(rng_seed: int | None) -> str:
      """Run a small neural SBI training loop and return the output directory path."""

In predator_prey_sbi/infer.py, define:
  def infer_from_file(config_path: str, observed_path: str) -> str:
      """Run the full inference pipeline and return the output directory path."""

In predator_prey_sbi/diagnostics.py, define:
  def posterior_predictive(
      posterior_samples: dict[str, list[float]],
      years: list[float],
      obs: tuple[list[float], list[float]],
      n_draws: int,
      rng_seed: int | None,
  ) -> dict[str, float]:
      """Return summary metrics and write plots comparing posterior simulations to observations."""

Use predator_prey_sbi/npe.py to train a neural posterior estimator and return a callable sampler; keep infer.py and diagnostics.py with stable signatures.

The CLI entry point should be implemented by adding if __name__ == "__main__" blocks in infer.py and diagnostics.py so users can run:
  python -m predator_prey_sbi.infer --config configs/base.yaml --observed data/LynxHare.txt
  python -m predator_prey_sbi.diagnostics --config configs/base.yaml --posterior runs/<timestamp>/posterior_samples.npz
  python -m predator_prey_sbi.smoke

Change Note: 2026-01-02 21:01Z — Updated the plan to commit to neural SBI only (sbi + torch), removed ABC references, added summary-statistics and neural posterior estimator interfaces, and introduced a neural SBI smoke-test step to align with the requested approach.
Change Note: 2026-01-03 06:31Z — Recorded the decision to use hand-crafted summary statistics instead of learned embeddings for the neural SBI inputs.
Change Note: 2026-01-03 06:31Z — Updated Progress to reflect completion of Milestone 1 implementation steps (data loader, simulator, CLI, config).
Change Note: 2026-01-04 17:35Z — Marked Milestone 2 complete, documented arviz home-directory permission issue, and recorded the decision to set HOME/MPLCONFIGDIR for SBI commands.
Change Note: 2026-01-04 18:53Z — Completed Milestone 3 by adding the inference pipeline, config wiring, and runtime helpers, and validated it with a successful inference run.
Change Note: 2026-01-04 19:02Z — Implemented diagnostics, added SBC and posterior predictive checks, and validated with a full diagnostics run.
