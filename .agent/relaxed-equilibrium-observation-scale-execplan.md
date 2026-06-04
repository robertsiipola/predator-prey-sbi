# Relax Equilibrium Priors With Observation Scaling

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository includes `.agent/PLANS.md` from the repository root. This ExecPlan must be maintained in accordance with `.agent/PLANS.md`.

## Purpose / Big Picture

The current best Lynx-Hare SBI configuration uses equilibrium-centered priors, which means the inferred latent hare and lynx equilibrium abundances are constrained near the observed data medians. Observation scaling means the simulator can infer separate multiplicative factors, `log_c_h` and `log_c_l`, that map latent abundance to observed fur-return scale. The January experiment found that observation scaling did not help while the equilibrium priors stayed tight; this plan tests the missing combination: wider equilibrium priors plus observation scaling.

After this change, a user can run a named experiment config, `configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml`, to train the same 2-round sequential SNPE baseline while allowing latent equilibria to range from 0.25x to 4.0x the observed medians and allowing observation scale factors to absorb differences between latent population scale and observed fur-return scale. Success is observable by running inference and diagnostics for this config and comparing RMSE, latent phase coherence, damping, and residual autocorrelation against the January sequential baseline.

## Progress

- [x] (2026-06-03 20:09Z) Created this focused ExecPlan after reading `.agent/PLANS.md` and confirming the existing code already supports observation scaling and explicit prior overrides.
- [x] (2026-06-03 20:09Z) Calculated relaxed equilibrium prior bounds from `data/LynxHare.txt`: hare median 40.97, lynx median 29.59, so 0.25x-to-4.0x log bounds are hare `[2.3265457304388892, 5.09913445267867]` and lynx `[2.0011421052922276, 4.773730827532009]`.
- [x] (2026-06-03 20:11Z) Added `configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml` and `tests/test_relaxed_eq_obs_scale_config.py`.
- [x] (2026-06-03 20:11Z) Ran `ruff format`, `ruff check`, `ty check`, and `pytest`; all checks passed with 22 tests.
- [x] (2026-06-04 17:44Z) Ran inference and diagnostics for `configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml`; posterior predictive RMSE was hare 36.47 and lynx 19.05, and latent-only diagnostics reported phase coherence ~0.30/0.20 with tau_damp p50 ~37.1y.

## Surprises & Discoveries

- Observation: No simulator or inference code change is required to start this experiment.
  Evidence: `predator_prey_sbi/priors.py` already accepts `include_observation_scale=True` and honors two-element prior overrides for `log_x_eq` and `log_y_eq`; `predator_prey_sbi/infer.py` and `predator_prey_sbi/diagnostics.py` pass those flags through when using the structure-aware prior scheme.
- Observation: The widened-prior observation-scale benchmark did not improve hare RMSE versus the January sequential baseline, although it improved lynx RMSE slightly.
  Evidence: `runs/2026-06-04_174331/diagnostics_metrics.json` reports hare RMSE 36.47 and lynx RMSE 19.05; `.agent/experiment_results.tsv` records the January `base_seq_4k.yaml` baseline at hare 35.56 and lynx 19.42.
- Observation: The wider equilibrium/scaling combination increased latent phase coherence relative to the January latent diagnostic but produced substantially damped latent draws.
  Evidence: `runs/2026-06-04_174356/diagnostics_metrics.json` reports latent phase coherence ~0.30/0.20 and tau_damp p50 ~37.1y; terminal diagnostics reported latent damping ratios ~0.55/0.56.

## Decision Log

- Decision: Implement the first milestone as a config-level experiment rather than a new prior mode.
  Rationale: The existing YAML `prior` override mechanism can express the relaxed bounds directly, and keeping the experiment explicit avoids adding a second abstraction before there is evidence that this combination improves inference.
  Date/Author: 2026-06-03, Codex

- Decision: Use 0.25x-to-4.0x observed-median equilibrium bounds for both species.
  Rationale: The previous structure-aware prior used 0.6x-to-1.6x, which may make observation scaling redundant. A 0.25x-to-4.0x range is wide enough to let latent abundance and observed fur returns decouple while still excluding implausibly tiny or huge equilibria.
  Date/Author: 2026-06-03, Codex

## Outcomes & Retrospective

The implementation and first benchmark are complete. The relaxed-equilibrium observation-scale experiment is available as a reusable config and is covered by a regression test. Its first 4k sequential benchmark does not beat the January baseline on hare RMSE, but it slightly improves lynx RMSE and phase coherence, so the result is a partial negative rather than a clear dead end.

## Context and Orientation

This repository implements simulation-based inference, or SBI, for predator-prey dynamics. SBI means the code simulates many synthetic Lynx-Hare time series under sampled model parameters, then trains a neural posterior estimator to infer which parameters are plausible for the observed `data/LynxHare.txt` time series.

The main experiment configuration path is `configs/experiments/`. Config files can contain `extends: ...` to inherit another YAML file and override selected fields. `configs/experiments/base_seq_4k.yaml` is the January best baseline: it extends `configs/base.yaml`, uses 4000 simulations, and runs two sequential SNPE rounds.

The relevant Python modules are:

- `predator_prey_sbi/config.py`, which loads YAML configs and recursively merges `extends` parents.
- `predator_prey_sbi/priors.py`, which builds the structure-aware prior ranges and parameter order from observed data and inference flags.
- `predator_prey_sbi/parameters.py`, which resolves inferred parameter dictionaries into simulator inputs.
- `predator_prey_sbi/npe.py`, which builds the simulator callable and trains the posterior.
- `predator_prey_sbi/infer.py`, which is the CLI entry point for training inference from a config.
- `predator_prey_sbi/diagnostics.py`, which runs posterior predictive, latent, residual, and SBC diagnostics.

Equilibrium abundance means the latent population level around which the model tends to cycle. In this code, `log_x_eq` is the log hare equilibrium and `log_y_eq` is the log lynx equilibrium. Observation scaling means multiplying latent abundance by an inferred observation factor before comparing it to observed data. In this code, `log_c_h` and `log_c_l` are log-scale observation factors for hare and lynx.

## Plan of Work

First, add `configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml`. It should extend `./base_seq_4k.yaml`, set `inference.include_observation_scale: true`, and override `inference.prior.log_x_eq` and `inference.prior.log_y_eq` with the relaxed bounds calculated from the observed medians. It can rely on the existing default `log_c_h` and `log_c_l` bounds of log 0.2 to log 5.0 from `build_structure_aware_prior`.

Second, add a regression test under `tests/` that loads the new config, loads `data/LynxHare.txt`, calls `build_structure_aware_prior` with the config flags and prior overrides, and proves the resulting parameter order contains `log_c_h` and `log_c_l` before the initial-condition offsets. The same test should assert that the prior bounds for `log_x_eq` and `log_y_eq` match the explicit relaxed values in the config.

Third, run the repository quality gates. The project uses `uv`, `ruff`, `ty`, and `pytest`. From the repository root, run:

    uv run ruff format .
    uv run ruff check .
    uv run ty check
    UV_CACHE_DIR=.uv-cache PYTHONPATH=. uv run pytest -q

The `UV_CACHE_DIR=.uv-cache` and `PYTHONPATH=.` form is used for pytest because this repository is not installed as a package during direct test collection, and uv may otherwise try to write to a restricted home cache.

Fourth, run a full benchmark once the config and tests pass:

    UV_CACHE_DIR=.uv-cache HOME=$PWD MPLCONFIGDIR=$PWD/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.infer --config configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml --observed data/LynxHare.txt

Then run diagnostics with the posterior path printed by inference:

    UV_CACHE_DIR=.uv-cache HOME=$PWD MPLCONFIGDIR=$PWD/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.diagnostics --config configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml --observed data/LynxHare.txt --posterior runs/<timestamp>/posterior_samples.npz

Latent-only diagnostics were run because lynx RMSE and phase coherence were still relevant despite worse hare RMSE.

## Concrete Steps

Work from `/Users/robertsiipola/predator-prey-sbi`.

Create the experiment config. The expected YAML is:

    extends: ./base_seq_4k.yaml

    inference:
      include_observation_scale: true
      prior:
        log_x_eq: [2.3265457304388892, 5.09913445267867]
        log_y_eq: [2.0011421052922276, 4.773730827532009]

Create or update tests so `UV_CACHE_DIR=.uv-cache PYTHONPATH=. uv run pytest -q` includes a test for this config and reports all tests passing.

After benchmarking, append one row to `.agent/experiment_results.tsv` with the timestamp, config path, posterior path, diagnostics JSON path, hare RMSE, and lynx RMSE. Update `.agent/sbi-execplan.md` with the result because it is the main long-running SBI plan.

## Validation and Acceptance

The implementation milestone is accepted when the new config exists, the new test proves the relaxed priors and observation-scale parameters are active, and all quality checks pass.

The experiment milestone is accepted because inference and diagnostics completed for `configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml`. The key comparison is against the January baseline row in `.agent/experiment_results.tsv` for `configs/experiments/base_seq_4k.yaml`, which had hare RMSE about 35.56 and lynx RMSE about 19.42. The relaxed-equilibrium observation-scale run produced hare RMSE 36.47 and lynx RMSE 19.05, so it should not replace the baseline. Its phase coherence improved versus the January latent diagnostic, but latent damping remained a concern.

## Idempotence and Recovery

Adding the config and tests is additive and safe to repeat. If an inference run fails before writing a posterior, rerun the same command. If diagnostics fail because the posterior path is wrong, use `find runs -maxdepth 2 -name posterior_samples.npz -print | sort | tail` to find the newest posterior and rerun diagnostics with that path. Do not delete prior runs; they are evidence for comparisons.

## Artifacts and Notes

The relaxed bounds were computed from `data/LynxHare.txt` as follows:

    hare median = 40.97
    lynx median = 29.59
    log(0.25 * hare median) = 2.3265457304388892
    log(4.0 * hare median) = 5.09913445267867
    log(0.25 * lynx median) = 2.0011421052922276
    log(4.0 * lynx median) = 4.773730827532009

## Interfaces and Dependencies

No new external dependencies are required. The existing `uv`, `ruff`, `ty`, `pytest`, `torch`, and `sbi` environment is sufficient.

The first milestone should leave these interfaces intact:

- `predator_prey_sbi.priors.build_structure_aware_prior(..., include_observation_scale=True)` returns a parameter order containing `log_c_h` and `log_c_l`.
- `predator_prey_sbi.config.load_config("configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml")` returns a merged config with `inference.include_observation_scale` set to `True` and the relaxed equilibrium prior bounds under `inference.prior`.

Change Note: 2026-06-03 20:09Z — Initial ExecPlan drafted and implementation started with a config-first design because existing prior override support is sufficient for the experiment.
Change Note: 2026-06-03 20:11Z — Added the relaxed-equilibrium observation-scale config and a regression test, then ran the required formatting, linting, typing, and pytest checks successfully.
Change Note: 2026-06-04 17:44Z — Completed the first full inference and diagnostics benchmark; recorded that the experiment worsened hare RMSE versus the January baseline while modestly improving lynx RMSE and latent phase coherence.
