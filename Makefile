.PHONY: all test lint format typecheck numbers paper

all: lint typecheck test numbers

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy cliffguard scripts

# Verify every number quoted in the paper, README and claims ledger still
# matches the measurement it came from.
numbers:
	uv run python scripts/check_paper_numbers.py

# Regenerate the paper's data, figures and tables, then compile it.
paper:
	uv run python scripts/build_paper_data.py
	uv run python scripts/build_paper_figures.py
	uv run python scripts/build_paper_tables.py
	cd docs/paper && latexmk -pdf cliff_artifact.tex
