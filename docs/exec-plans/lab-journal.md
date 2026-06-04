# Lab Journal

This journal is the narrative companion to `docs/exec-plans/lab-journal.tsv`. Use it to summarize scientific decisions, negative results, and why a modeling branch should continue or stop.

The TSV table remains the source of truth for benchmark rows with these columns:

    timestamp_utc	config	posterior	diagnostics	rmse_hare	rmse_lynx

## Current Baseline

The strongest recorded hare RMSE remains `configs/experiments/base_seq_4k.yaml` from 2026-01-09, with hare RMSE about 35.56 and lynx RMSE about 19.42.

## Recent Results

The correlated process / AR(1) observation residual experiment improved latent phase coherence but worsened hare RMSE and damping, so it should not receive larger simulation budgets without tighter priors.

The relaxed-equilibrium observation-scale experiment slightly improved lynx RMSE and phase coherence but worsened hare RMSE, so it should not replace the January sequential baseline.
