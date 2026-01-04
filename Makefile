.PHONY: format lint typecheck check simulate smoke infer

format:
	ruff format .

lint:
	ruff check .

typecheck:
	ty check .

check: format lint typecheck

simulate:
	mkdir -p .cache/matplotlib
	MPLCONFIGDIR=$(PWD)/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.simulate --config configs/base.yaml

smoke:
	mkdir -p .cache/matplotlib arviz_data
	HOME=$(PWD) MPLCONFIGDIR=$(PWD)/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.smoke

infer:
	mkdir -p .cache/matplotlib arviz_data
	HOME=$(PWD) MPLCONFIGDIR=$(PWD)/.cache/matplotlib uv run --no-cache python -m predator_prey_sbi.infer --config configs/base.yaml --observed data/LynxHare.txt
