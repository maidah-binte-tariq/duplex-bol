# Thin wrappers over uv so "what do I run" is never a question.
# `make check` is the gate CI enforces; run it before you push.

.DEFAULT_GOAL := help
PY := uv run

.PHONY: help setup test lint format typecheck check demo demo-corpus figures bench clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install dev + moshi extras
	uv venv
	uv pip install -e ".[dev,moshi]"
	$(PY) pre-commit install

test: ## Run the test suite with coverage
	$(PY) pytest --cov --cov-report=term-missing

lint: ## Ruff lint
	$(PY) ruff check src tests scripts

format: ## Ruff auto-format
	$(PY) ruff format src tests scripts

typecheck: ## mypy (strict) on the package
	$(PY) mypy src

check: lint typecheck test ## Everything CI runs
	$(PY) ruff format --check src tests scripts

demo: ## Run the fake cascade end-to-end and print the latency budget
	$(PY) duplex-bol demo

demo-corpus: ## Generate a tiny synthetic corpus under data/demo/
	$(PY) python scripts/make_demo_corpus.py --out data/demo

figures: ## Regenerate docs/assets/ diagrams from live code
	uv pip install -q matplotlib && $(PY) python scripts/make_figures.py

bench: ## Run benchmarks (tokenizer fertility + latency) -> tables, results.json, figure
	uv pip install -q matplotlib && $(PY) python scripts/run_benchmarks.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
