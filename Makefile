.PHONY: install install-dev test format lint clean

# Install the package
install:
	uv pip install -e .

# Install the package with development dependencies
install-dev:
	uv pip install -e ".[dev]"

# Run tests
test:
	pytest tests/

# Run tests with coverage
test-cov:
	pytest --cov=src/agentic_rag tests/

# Format code
format:
	black src tests
	isort src tests

# Lint code
lint:
	ruff check src tests
	mypy src tests

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete