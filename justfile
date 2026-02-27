# depth-view command runner

# Install all dependencies
install:
    uv sync --dev

# Launch Gradio demo
demo *ARGS:
    uv run python -m depthview.serving.gradio_app {{ARGS}}

# Launch FastAPI server
serve *ARGS:
    uv run python -m depthview.serving.api {{ARGS}}

# Run all tests
test *ARGS:
    uv run pytest tests/ {{ARGS}}

# Run tests with coverage
test-cov:
    uv run pytest tests/ --cov=depthview --cov-report=term-missing

# Run fast tests only (skip model downloads)
test-fast:
    uv run pytest tests/ -m "not slow"

# Lint code
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Auto-fix lint + format
fix:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/

# All quality checks
check: lint test-fast
