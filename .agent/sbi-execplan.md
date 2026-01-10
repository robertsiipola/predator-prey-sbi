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
- [x] (2026-01-05 12:35Z) Reparameterized LV and anchored initial conditions, adding shared parameter-resolution helpers and updating configs/defaults.
- [x] (2026-01-05 12:44Z) Added logistic prey growth via carrying capacity k and updated priors/configs to include it.
- [x] (2026-01-05 12:45Z) Ran inference + diagnostics with logistic prey growth; recorded improved RMSE and SBC.
- [x] (2026-01-05 12:45Z) Ran inference + diagnostics with reparameterized posterior; recorded RMSE and SBC results for comparison.
- [x] (2026-01-05 12:55Z) Updated diagnostics loader to accept reparameterized posterior samples.
- [x] (2026-01-05 14:55Z) Added inferred observation noise parameters (sigma_h, sigma_l) and updated simulator wiring/configs.
- [x] (2026-01-05 14:55Z) Ran inference + diagnostics with inferred observation noise; recorded RMSE and SBC results.
- [x] (2026-01-05 15:06Z) Added mechanistic regression summaries derived from log-growth vs. predator/prey levels.
- [x] (2026-01-05 15:13Z) Ran inference + diagnostics with mechanistic regression summaries; recorded RMSE and SBC.
- [x] (2026-01-05 15:47Z) Added learned embedding pathway for full time-series inputs (CNN + diffs) and updated config defaults.
- [x] (2026-01-05 15:47Z) Ran inference + diagnostics with learned embedding; recorded RMSE and SBC results.
- [x] (2026-01-05 19:10Z) Added structure-aware parameterization with data-informed priors (log_T/log_r/log_x_eq/log_y_eq/log_k_ratio).
- [x] (2026-01-05 19:11Z) Ran inference + diagnostics with structure-aware priors; recorded RMSE and SBC results.
- [x] (2026-01-05 19:23Z) Tightened structure-aware priors (T/r/x_eq/y_eq/k_ratio, eps, sigma) and reran inference + diagnostics; recorded RMSE and SBC results.
- [x] (2026-01-05 19:29Z) Increased structure-aware simulation budget to 4k and reran inference + diagnostics; recorded RMSE and SBC results.
- [x] (2026-01-09 17:31Z) Added annual-mean and midpoint observation operators, posterior predictive mean RMSE reporting, and a latent process-noise option (log_sigma_p); ran an experiment sweep and recorded results in .agent/experiment_results.tsv.
- [x] (2026-01-09 17:33Z) Added optional Holling type-II predation and discovered/fixed a bug where predator growth mistakenly used delta*(beta*predation) instead of delta*predation.
- [x] (2026-01-09 18:05Z) Increased simulation budget to 4k for the current best-performing baseline and achieved hare RMSE ~35.6 (lynx ~19.8) on posterior predictive mean.
- [ ] (2026-01-09 18:57Z) Next: scale up simulation budget sweep (completed: 6k/8k; remaining: 12k) for the current best config and record the RMSE vs. runtime curve.
- [x] (2026-01-09 20:31Z) Tested a process-noise prior upper bound increase (to log(0.6)) while capping observation noise (to log(0.2)); it degraded RMSE in this run.
- [x] (2026-01-09 20:31Z) Evaluated posterior sampling method (rejection vs MCMC): MCMC slightly improved one 4k run but sequential+MCMC was slow and degraded RMSE.
- [x] (2026-01-09 22:48Z) Implemented and tested a 2-round SNPE schedule with a 20% prior mix-in; it produced a small RMSE improvement versus the single-round baseline in this run.
- [x] (2026-01-10 07:33Z) Added a latent posterior-draw diagnostic (to separate damping vs phase decoherence) and discovered that mean-flattening is primarily phase decoherence: per-draw oscillations persist (median damping ratio ~1.05–1.08) but phase coherence is low (~0.16–0.18); also found config experiments were not inheriting from configs/base.yaml until adding an `extends:` mechanism.
- [x] (2026-01-10 07:43Z) Implemented an inferred fractional observation lag (obs_lag in [0,1) years) and wired it through simulator/inference/diagnostics; first run did not materially change phase coherence or RMSE but provides the knob needed to test phase-alignment hypotheses.
- [x] (2026-01-10 13:32Z) Prototyped an equilibrium-centered observation power index (p) (fur returns as a nonlinear index of abundance) and ran configs/experiments/obs_power.yaml; it produced posterior predictive RMSE hare 38.30 (lynx 19.21) and increased phase coherence (~0.32/0.27) but tended to reintroduce damping (tau_damp p50 ~34y).
- [ ] (2026-01-09 18:57Z) Next: revisit observation scaling (log_c_h/log_c_l) with relaxed equilibrium priors (widen log_x_eq/log_y_eq ranges), since the current equilibrium-anchored priors may make scaling redundant.

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
- Observation: Adding inferred initial conditions (hare0/lynx0) and richer summaries did not reduce RMSE in the first trial (hare ~58, lynx ~32).
  Evidence: diagnostics_metrics.json in runs/2026-01-04_211553.
- Observation: Diagnostics initially failed on reparameterized posterior samples because the loader required beta/delta keys.
  Evidence: ValueError "Posterior samples missing keys: ['beta', 'delta']" when running diagnostics on runs/2026-01-05_123432.
- Observation: Reparameterized LV with anchored initial conditions did not improve RMSE in the first run (hare ~50.5, lynx ~23.7).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_123555.
- Observation: Adding logistic prey growth (carrying capacity k) materially improved RMSE (hare ~35.3, lynx ~19.9).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_124403.
- Observation: Inferring observation noise (sigma_h, sigma_l) kept hare RMSE similar while improving lynx RMSE slightly (hare ~35.3, lynx ~18.7).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_145536.
- Observation: Mechanistic regression summaries increased the summary length to include slope/intercept/R2 for log-growth regressions.
  Evidence: summarize_series now appends six regression features.
- Observation: Mechanistic regression summaries degraded RMSE versus the prior run (hare ~38.4, lynx ~20.2).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_151311.
- Observation: Learned embedding (CNN on log1p + diffs) did not improve RMSE in the first run (hare ~38.5, lynx ~19.8).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_154708.
- Observation: Structure-aware priors and parameterization did not improve RMSE in the first run (hare ~37.9, lynx ~19.9).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_191017.
- Observation: Tightening structure-aware priors slightly improved RMSE but still trails the best prior run (hare ~37.3, lynx ~19.6).
  Evidence: diagnostics_metrics.json in runs/2026-01-05_192244.
- Observation: A 4k simulation budget with tightened structure-aware priors further improved RMSE (hare ~36.5, lynx ~18.8) but still trails the best prior run.
  Evidence: diagnostics_metrics.json in runs/2026-01-05_192911.
- Observation: Adding annual-mean observation operator and latent process noise provides small but consistent RMSE improvements; the best improvement in this round came from increasing the simulation budget to 4k.
  Evidence: runs/2026-01-09_180528/diagnostics_metrics.json reports posterior predictive mean RMSE hare=35.587, lynx=19.797 using configs/experiments/base_4k.yaml.
- Observation: Adding per-species observation scaling parameters (log_c_h/log_c_l) did not improve RMSE under the current structure-aware priors (x_eq/y_eq anchored to data medians).
  Evidence: .agent/experiment_results.tsv shows worse hare RMSE for configs/experiments/linear_obs_scale.yaml vs configs/base.yaml on 2026-01-09.
- Observation: Rejection sampling can become extremely slow when the learned posterior is narrow (very low acceptance), which can dominate experiment runtime.
  Evidence: sbi emitted a low acceptance warning during posterior sampling in a prior sweep (warning reported ~0.6% acceptance).
- Observation: 2-round sequential SNPE (with a 20% prior mix-in) provided a small RMSE improvement at fixed budget, but MCMC posterior sampling was extremely slow and degraded RMSE in the tested sequential run.
  Evidence: .agent/experiment_results.tsv shows hare RMSE ~35.56 for configs/experiments/base_seq_4k.yaml and ~37.22 for configs/experiments/base_seq_4k_mcmc.yaml on 2026-01-09.
- Observation: Under the current best sequential config, individual posterior draws keep oscillating but the pointwise mean flattens due to low phase coherence across draws.
  Evidence: runs/2026-01-10_073235/diagnostics_metrics.json reports latent_phase_coherence_hare≈0.18 and latent_phase_coherence_lynx≈0.16 with median latent_damping_ratio_hare≈1.08 and latent_damping_ratio_lynx≈1.05 (plot: runs/2026-01-10_073235/latent_posterior_draws.png).
- Observation: The inferred damping timescale is often shorter than the full record length, even though per-draw oscillations can persist via process noise; ~90% of draws have tau_damp < record length in the diagnostic run.
  Evidence: runs/2026-01-10_073235/diagnostics_metrics.json reports tau_damp_frac_lt_record≈0.895 with tau_damp_years_p50≈35.3 years.
- Observation: Adding an inferred observation lag parameter did not substantially increase phase coherence in the first run; coherence remained low (~0.18) and RMSE stayed ~36.
  Evidence: runs/2026-01-10_074242/diagnostics_metrics.json reports latent_phase_coherence_hare≈0.19 and latent_phase_coherence_lynx≈0.18 with posterior_predictive_hare_rmse≈36.28 (config: configs/experiments/base_seq_4k_lag_latent_diag.yaml).

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
- Decision: Reparameterize Lotka-Volterra using (alpha, gamma, x_star, y_star) and derive beta=alpha/y_star and delta=gamma/x_star; anchor initial conditions via eps_h0/eps_l0 around the first observation.
  Rationale: This reduces identifiability issues and keeps initial conditions from absorbing dynamics mismatch while still allowing modest adjustments.
  Date/Author: 2026-01-05, Codex
- Decision: Add logistic prey growth with carrying capacity k and infer it alongside other parameters.
  Rationale: The deterministic LV model underfits amplitude regulation; a carrying capacity is the smallest structural change likely to reduce RMSE.
  Date/Author: 2026-01-05, Codex
- Decision: Infer observation noise (sigma_h, sigma_l) on the log scale instead of fixing noise_scale.
  Rationale: Allowing separate observation noise per series reduces pressure on dynamics parameters to explain variability.
  Date/Author: 2026-01-05, Codex
- Decision: Add mechanistic regression summaries based on log-growth vs. predator/prey levels.
  Rationale: These features directly target the LV structure and can improve identifiability without a learned embedding.
  Date/Author: 2026-01-05, Codex
- Decision: Add a learned embedding on the full log1p time series with differences using a small CNN and structured z-scoring.
  Rationale: Summary-stat inputs may be brittle; an embedding can capture phase and amplitude information directly.
  Date/Author: 2026-01-05, Codex
- Decision: Introduce structure-aware parameters (log_T, log_r, log_x_eq, log_y_eq, log_k_ratio) with data-informed priors.
  Rationale: Enforcing K > x_eq and using interpretable scale priors should reduce wasted simulations and improve identifiability.
  Date/Author: 2026-01-05, Codex
- Decision: Tighten structure-aware priors to focus on plausible cycle periods (8–14 years), equilibria near the data medians, and smaller observation noise.
  Rationale: Narrower, data-informed priors reduce wasted simulations and stabilize the embedding network without over-constraining dynamics.
  Date/Author: 2026-01-05, Codex
- Decision: Assume the second column in data/LynxHare.txt is the prey (hare) series and the third column is the predator (lynx) series, with units treated as relative counts.
  Rationale: This is the common ordering for the lynx-hare dataset; the plan remains flexible if a different ordering is confirmed.
  Date/Author: 2026-01-02, Codex
- Decision: Add a latent process noise term (sigma_p) applied once per year between ODE integration steps, and infer it as log_sigma_p.
  Rationale: Deterministic logistic LV tends to produce overly-regular or damped oscillations; process noise can sustain variability and amplitude fluctuations that observation noise cannot.
  Date/Author: 2026-01-09, Codex
- Decision: Add observation operators (point, midpoint, annual_mean) for mapping continuous-time latent states to annual observations.
  Rationale: Annual sampling can induce systematic phase mismatch; midpoint/annual averaging is a low-cost way to reduce aliasing.
  Date/Author: 2026-01-09, Codex
- Decision: Add optional Holling type-II predation (with inferred log_h) as a minimal structural extension beyond linear predation.
  Rationale: Saturating predation is a common mechanism for stabilizing cycles and may better match amplitude regulation in the lynx-hare record.
  Date/Author: 2026-01-09, Codex
- Decision: Implement 2-round sequential SNPE (with optional prior mix-in) as an inference option, and force rejection sampling for proposal draws even when final posterior samples use MCMC.
  Rationale: Sequential SNPE can concentrate simulations near the observed dataset, but proposal draws must be fast; MCMC proposal sampling was prohibitively slow in practice.
  Date/Author: 2026-01-09, Codex
- Decision: Add a `extends:` mechanism to YAML config loading and update all experiment configs to extend configs/base.yaml.
  Rationale: Many experiment YAMLs were intended as overrides of configs/base.yaml but previously only overrode DEFAULT_CONFIG, silently changing observation operator and inference toggles (e.g., annual_mean and include_process_noise).
  Date/Author: 2026-01-10, Codex
- Decision: Add a latent posterior-draw diagnostic (overlay of latent trajectories + phase/damping summaries) to guide whether to change dynamics or observation timing.
  Rationale: RMSE and mean trajectories can be misleading when posterior draws are oscillatory but out of phase; separating damping vs phase decoherence determines the next modeling move.
  Date/Author: 2026-01-10, Codex
- Decision: Add an inferred fractional observation lag parameter (obs_lag) and apply it consistently across all observation operators (point/midpoint/annual_mean), including the process-noise path.
  Rationale: If annual sampling is systematically phase-shifted relative to the underlying ecological dynamics (or fur-return timing), a shared fractional lag is a minimal, testable way to improve alignment without changing the ecological model.
  Date/Author: 2026-01-10, Codex

## Outcomes & Retrospective

Milestones 1–4 are implemented and verified with inference + diagnostics runs. Logistic prey growth (carrying capacity k) improved RMSE substantially versus the baseline. Adding inferred observation noise marginally improved lynx RMSE while keeping hare RMSE similar, while mechanistic regression summaries, the learned embedding, and the structure-aware priors (even after tightening) did not improve RMSE in the first trials; further gains likely need a larger simulation budget, alternative priors, or model/observation adjustments.

As of 2026-01-09, the best observed hare RMSE in this round of experiments is ~35.6 (lynx ~19.4) using annual_mean observations, inferred latent process noise (log_sigma_p), and a 2-round sequential SNPE run with 4k total simulations (see configs/experiments/base_seq_4k.yaml). Additional structural extensions (Holling type-II) did not improve RMSE in the first sweep, and per-species observation scaling did not help under the current equilibrium-anchored priors.

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

To minimize posterior predictive RMSE when the baseline model underfits, prioritize the following improvement options in order of expected impact. First, extend the simulator to include logistic prey growth (a carrying capacity parameter) so the model can reproduce amplitude regulation; this adds one parameter but typically improves both phase and amplitude fit. Second, add process noise (stochastic LV) or infer observation noise to capture irregular cycle amplitudes; this often reduces RMSE at the cost of a slightly broader posterior. Third, infer initial conditions (hare0/lynx0) and optional scaling factors for observed counts to align simulated amplitude with data; this helps when the oscillation scale is mismatched. Fourth, increase the simulation budget or move to multi-round SNPE to reduce estimator bias once the model is expressive enough. Finally, refine summary statistics (e.g., period from autocorrelation, phase lag, peak amplitude ratios) if the posterior predictive plot suggests phase or amplitude mismatches despite a good average fit.

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
Change Note: 2026-01-05 11:24Z — Added a ranked list of RMSE-improvement options (model extensions, noise, initial conditions, simulation budget, summary refinement) and recorded that initial-condition inference did not reduce RMSE in the first trial.
Change Note: 2026-01-05 11:24Z — Added a decision to reparameterize LV using (alpha, gamma, x_star, y_star) and anchor initial conditions via epsilons around the first observation.
Change Note: 2026-01-05 12:55Z — Recorded reparameterization experiment results, diagnostics loader fix, and updated progress/outcomes to reflect current state.
Change Note: 2026-01-05 12:45Z — Added logistic prey growth with carrying capacity k and recorded the improved RMSE results.
Change Note: 2026-01-05 14:55Z — Added inferred observation noise (sigma_h, sigma_l) and recorded the updated RMSE/SBC results.
Change Note: 2026-01-05 15:06Z — Added mechanistic regression summaries to the feature set.
Change Note: 2026-01-05 15:13Z — Recorded mechanistic summary experiment results (RMSE and SBC).
Change Note: 2026-01-05 15:47Z — Added learned embedding pathway and recorded the initial embedding experiment results.
Change Note: 2026-01-05 19:11Z — Added structure-aware parameterization/priors and recorded the initial results.
Change Note: 2026-01-09 17:31Z — Added observation operators (point/midpoint/annual_mean), latent process noise (log_sigma_p), tau_damp/k_ratio switch, and posterior predictive mean RMSE; ran a sweep and recorded results in .agent/experiment_results.tsv.
Change Note: 2026-01-09 17:33Z — Added optional Holling type-II predation and fixed a Holling/linear predation bug in predator growth term.
Change Note: 2026-01-09 18:05Z — Increased simulation budget experiments to 4k and recorded the improved RMSE for configs/experiments/base_4k.yaml.
Change Note: 2026-01-09 23:40Z — Added optional 2-round sequential SNPE (with prior mix-in) and new experiment configs/scripts; recorded updated RMSE results in .agent/experiment_results.tsv.
