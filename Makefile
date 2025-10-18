.PHONY: help install install-dev clean build test lint format check-format publish-test publish dist

# Default target
help:
	@echo "Available commands:"
	@echo "  install	  - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  clean		- Clean build artifacts"
	@echo "  build		- Build package distributions"
	@echo "  test		 - Run tests"
	@echo "  lint		 - Run linting checks"
	@echo "  format	   - Format code"
	@echo "  check-format - Check code formatting"
	@echo "  publish-test - Publish to Test PyPI"
	@echo "  publish	  - Publish to PyPI"
	@echo "  dist		 - Build and show distribution info"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-all:
	pip install -e .[all]

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

test:
	pytest -q

test-verbose:
	pytest -v

test-coverage:
	pytest --cov=kerb --cov-report=html

lint:
	flake8 kerb tests
	pylint kerb

lint-fix:
	black kerb tests
	isort kerb tests
	@echo "Auto-fixable issues have been resolved. Running lint check..."
	flake8 kerb tests
	pylint kerb

# Alternative: separate auto-fix targets
autofix:
	black kerb tests
	isort kerb tests

autopep8-fix:
	autopep8 --in-place --recursive kerb tests

format:
	black kerb tests
	isort kerb tests

check-format:
	black --check kerb tests
	isort --check-only kerb tests

publish-test: build
	python -m twine upload --verbose --repository testpypi dist/*

publish: build
	python -m twine upload --verbose dist/*

# Distribution info
dist: build
	@echo "Distribution files created:"
	@ls -la dist/
	@echo "\nPackage contents:"
	@python -m tarfile -l dist/*.tar.gz

# Development setup
setup-dev:
	pip install build twine pytest black flake8 pylint isort

# Check if package can be imported
check-import:
	python -c "import kerb; print('✓ Package imports successfully')"

# Version bump helpers
version-patch:
	@echo "Current version: $$(grep version pyproject.toml)"
	@echo "Remember to update version in pyproject.toml for patch release"

version-minor:
	@echo "Current version: $$(grep version pyproject.toml)"
	@echo "Remember to update version in pyproject.toml for minor release"

version-major:
	@echo "Current version: $$(grep version pyproject.toml)"
	@echo "Remember to update version in pyproject.toml for major release"
