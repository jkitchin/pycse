# CLAUDE.md - Project Guide for AI Assistants

This file provides context for Claude and other AI assistants working with this codebase.

## Project Overview

**pycse** (Python Computations in Science and Engineering) is a Python library for scientific computing and engineering calculations. It provides:

- Regression and curve fitting with confidence intervals
- ODE solving with units support
- Machine learning models (sklearn-compatible)
- JAX-based neural networks
- Design of experiments utilities

**Version**: 2.7.0
**Python**: >=3.9
**License**: GPL-3.0-or-later

## Directory Structure

```
pycse/
├── src/pycse/           # Main source code
│   ├── PYCSE.py         # Core regression functions (regress, nlinfit, etc.)
│   ├── beginner.py      # Beginner-friendly utilities
│   ├── utils.py         # General utilities
│   ├── sklearn/         # sklearn-compatible ML models
│   │   ├── kan.py       # Kolmogorov-Arnold Networks
│   │   ├── jax_*.py     # JAX-based neural networks
│   │   ├── dpose.py     # Deep ensemble models
│   │   └── ...
│   ├── tests/           # All tests live here
│   └── examples/        # Example notebooks
├── pycse-jb/            # Jupyter Book documentation (primary docs)
├── pycse-channel/       # YouTube companion materials
├── archive/             # Legacy org-mode documentation
├── data/                # Example data files
├── docker/              # Docker development environment
└── .github/workflows/   # CI configuration
```

## Development Setup

```bash
# Create virtual environment and install
uv venv
uv pip install -e ".[test]"

# Or with pip
pip install -e ".[test]"
```

## Testing

See `TESTING.md` for comprehensive testing documentation.

### Quick Commands

```bash
# Fast tests only (recommended for development, ~10 seconds)
pytest -m "not slow"

# All tests
pytest

# Slow tests only (ML/training, ~40 minutes)
pytest -m "slow"

# With coverage
pytest -m "not slow" --cov=src/pycse --cov-report=term-missing

# Parallel execution
pytest -m "not slow" -n auto

# Specific test file
pytest src/pycse/tests/test_pycse.py -v
```

### Test Markers

- `@pytest.mark.slow` - ML/training tests (>1 second)
- Use `pytestmark = pytest.mark.slow` for entire modules

### CI Workflows

- **pycse-tests.yaml** - Fast tests on every push/PR (~2 min)
- **pycse-tests-slow.yaml** - Slow tests on master/nightly (~40 min)

## Code Style

- **Linter**: Ruff (configured in pyproject.toml)
- **Line length**: 100 characters
- **Target**: Python 3.9+

```bash
# Run linter
ruff check src/pycse

# Fix auto-fixable issues
ruff check --fix src/pycse
```

## Key Modules

### Core (`src/pycse/PYCSE.py`)
- `regress()` - Linear regression with confidence intervals
- `nlinfit()` - Nonlinear curve fitting
- `odelay()` - ODE solving with event detection

### sklearn Models (`src/pycse/sklearn/`)
All models follow sklearn's estimator API (fit/predict/score).

- `KANRegressor` - Kolmogorov-Arnold Networks
- `MonotonicNN` - Monotonic neural networks (JAX)
- `PeriodicNN` - Periodic neural networks (JAX)
- `ICNN` - Input-convex neural networks (JAX)
- `DPOSE` - Deep ensemble with uncertainty
- `KFoldNN` - K-fold neural network ensemble

### Utilities
- `hashcache.py` - Disk caching for expensive computations
- `beginner.py` - Simplified interfaces for teaching

## Important Notes

### When Modifying Tests
- Mark slow tests (training, >1 sec) with `@pytest.mark.slow`
- Use small datasets (30-50 samples) for fast tests
- Use few epochs (5-10) for training tests
- See TESTING.md for detailed guidelines

### When Adding sklearn Models
- Follow sklearn estimator conventions
- Implement `fit()`, `predict()`, `score()`
- Add `get_params()` and `set_params()` for hyperparameters
- Include tests in `src/pycse/sklearn/tests/`

### Pre-commit Hooks
Pre-commit runs fast tests only. Ensure `pytest -m "not slow"` passes before committing.

## Git Conventions

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `test:`
- Never use `--no-verify` with git commands
- Run fast tests before committing

## Building and Publishing

```bash
uv build      # Build wheel and sdist
uv publish    # Publish to PyPI
```

## Resources

- **Documentation**: `pycse-jb/` (Jupyter Book)
- **Testing Guide**: `TESTING.md`
- **PyPI**: https://pypi.org/project/pycse/
- **GitHub**: https://github.com/jkitchin/pycse
