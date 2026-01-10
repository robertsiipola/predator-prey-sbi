#!/usr/bin/env zsh
set -euo pipefail

echo "Running targeted experiments in $(pwd)"

out_file=".agent/experiment_results.tsv"
if [[ ! -f "$out_file" ]]; then
  echo -e "timestamp_utc\tconfig\tposterior\tdiagnostics\trmse_hare\trmse_lynx" > "$out_file"
fi

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

typeset -a CONFIGS
CONFIGS=(
  configs/experiments/base_4k.yaml
  configs/experiments/base_seq_4k.yaml
  configs/experiments/base_4k_mcmc.yaml
  configs/experiments/base_4k_sigma_p_hi_obs_cap02.yaml
  configs/experiments/base_6k.yaml
  configs/experiments/base_8k.yaml
)

for cfg in "${CONFIGS[@]}"; do
  post=$(HOME=$PWD MPLCONFIGDIR=$PWD/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.infer --config "$cfg" --observed data/LynxHare.txt | rg -o "runs/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}/posterior_samples\\.npz" | tail -n 1)
  diag_json=$(HOME=$PWD MPLCONFIGDIR=$PWD/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.diagnostics --config "$cfg" --observed data/LynxHare.txt --posterior "$post" | rg -o "runs/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}/diagnostics_metrics\\.json" | head -n 1)
  rmse_line=$(uv run --no-cache python - <<PY
import json
from pathlib import Path
p=Path("$diag_json")
m=json.loads(p.read_text())
print(f"{m.get('posterior_predictive_hare_rmse')}\t{m.get('posterior_predictive_lynx_rmse')}")
PY
)
  echo -e "${ts}\t${cfg}\t${post}\t${diag_json}\t${rmse_line}" >> "$out_file"
done

echo "Appended results to $out_file"
