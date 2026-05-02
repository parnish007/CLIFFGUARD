.PHONY: all test lint format typecheck

all: lint typecheck test

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy cliffguard
