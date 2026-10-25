# List available tasks.
default:
    @just --list

# Install locked dependencies with a PyTorch build (cpu, cuda or rocm) and the pre-commit hooks.
setup build="cpu":
    uv sync --locked --extra {{build}}
    uv run --no-sync pre-commit install

# Format Python code.
format:
    uv run ruff format .

# Check formatting and lint rules.
lint:
    uv run ruff format --check .
    uv run ruff check .

# Check types.
typecheck:
    uv run ty check

# Check that tests mirror src/contcheck/ and the skills link is intact.
layout:
    uv run python scripts/check_layout.py

# Run the default test suite (no downloads, no LLM, no datasets).
test:
    uv run pytest

# Run every test, including those that download models or datasets or need an LLM.
test-all:
    uv run pytest -m ""

# Run the test suite with branch coverage (writes coverage.xml).
coverage:
    uv run pytest --cov --cov-report=term --cov-report=xml

# Run all required checks.
check: lint typecheck layout test

# Run the CLI (pass arguments after the recipe name).
cli *args:
    uv run contcheck {{args}}

# Render design-docs/images/*.dot to PNG next to each source (needs Graphviz).
diagrams:
    uv run python -c "import pathlib, subprocess; [subprocess.run(['dot', '-Tpng', '-Gdpi=110', str(f), '-o', str(f.with_suffix('.png'))], check=True) for f in sorted(pathlib.Path('design-docs/images').glob('*.dot'))]"
