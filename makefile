.PHONY: help test test-verbose test-cov test-cov-html clean install lint format check

help:
	@echo "pycse Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Install package with test dependencies"
	@echo "  test            Run all tests"
	@echo "  test-verbose    Run tests with verbose output"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  test-cov-html   Run tests with HTML coverage report"
	@echo "  test-file       Run a specific test file (use FILE=path/to/test.py)"
	@echo "  test-func       Run a specific test function (use FUNC=test_name)"
	@echo "  lint            Run ruff linter"
	@echo "  format          Format code with ruff"
	@echo "  check           Run ruff check without fixing"
	@echo "  clean           Clean up temporary files and caches"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make test-cov-html"
	@echo "  make test-file FILE=src/pycse/tests/test_pycse.py"
	@echo "  make test-func FUNC=test_regress"

install:
	uv pip install -e ".[test]"

test:
	pytest src/pycse/tests

test-verbose:
	pytest src/pycse/tests -vv

test-cov:
	pytest src/pycse/tests --cov=src/pycse --cov-report=term-missing

test-cov-html:
	pytest src/pycse/tests --cov=src/pycse --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html in your browser"

test-file:
ifndef FILE
	@echo "Error: FILE not specified"
	@echo "Usage: make test-file FILE=src/pycse/tests/test_pycse.py"
	@exit 1
endif
	pytest $$(FILE) -vv

test-func:
ifndef FUNC
	@echo "Error: FUNC not specified"
	@echo "Usage: make test-func FUNC=test_regress"
	@exit 1
endif
	pytest src/pycse/tests -k $$(FUNC) -vv

test-not-slow:
	pytest src/pycse/tests -m "not slow"

test-regression:
	pytest src/pycse/tests -m regression

lint:
	ruff check src/pycse --fix

format:
	ruff format src/pycse

check:
	ruff check src/pycse

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up temporary files"

# CI/CD targets
ci-test:
	pytest src/pycse/tests -v --cov=src/pycse --cov-report=term-missing --cov-report=xml

# Development shortcuts
dev-setup: install
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

watch-tests:
	@echo "Watching for changes and running tests..."
	@echo "Press Ctrl+C to stop"
	@while true; do \
		pytest src/pycse/tests --exitfirst --quiet; \
		sleep 2; \
	done
