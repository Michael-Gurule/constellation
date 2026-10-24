# Developer entry points. Run inside the activated conda env:
#   conda activate constellation
PYTHON ?= python
ENV_NAME ?= constellation

.DEFAULT_GOAL := help
.PHONY: help env env-update env-collector hooks lint format typecheck test cov bench check collect-iss data-benchmarks clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

# On macOS the torch wheel bundles its own OpenMP runtime, and LightGBM and
# XGBoost load the conda one. Two runtimes in one process crash during
# training, so torch is pointed at the conda copy.
define LINK_OPENMP
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		prefix=$$(conda run -n $(ENV_NAME) python -c 'import sys; print(sys.prefix)'); \
		ln -sf "$$prefix/lib/libomp.dylib" "$$prefix/lib/python3.12/site-packages/torch/lib/libomp.dylib"; \
		echo "Linked torch to the shared OpenMP runtime"; \
	fi
endef

env:  ## Create the conda environment from environment.yml
	conda env create -f environment.yml
	$(LINK_OPENMP)

env-update:  ## Sync the conda environment with environment.yml
	conda env update -n $(ENV_NAME) -f environment.yml --prune
	$(LINK_OPENMP)

env-collector:  ## Create the collector-only conda env (no PyTorch; for Intel Macs)
	conda create -y -n constellation-collector --override-channels -c conda-forge python=3.12 pip
	conda run -n constellation-collector python -m pip install -r requirements-collector.txt
	conda run -n constellation-collector python -m pip install --no-deps -e .

hooks:  ## Install pre-commit git hooks
	$(PYTHON) -m pre_commit install

lint:  ## Lint and check formatting (no changes)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-fix lint findings and format code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:  ## Static type check (mypy, strict with legacy ratchet)
	$(PYTHON) -m mypy

test:  ## Run the test suite
	MPLBACKEND=Agg $(PYTHON) -m pytest

cov:  ## Run tests with coverage
	MPLBACKEND=Agg $(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml

bench:  ## Latency benchmarks with budgets (benchmarks/)
	MPLBACKEND=Agg $(PYTHON) -m pytest benchmarks --benchmark-only --benchmark-columns=min,median,max,rounds

check: lint typecheck cov bench  ## Everything CI runs

collect-iss:  ## Collect ISS telemetry until stopped (see scripts/collect_iss.sh)
	scripts/collect_iss.sh

data-benchmarks:  ## Fetch the labeled benchmarks (large: ESA-Mission1 is 3.8 GB)
	constellation data fetch --all

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -not -path "./.git/*" -prune -exec rm -rf {} +
