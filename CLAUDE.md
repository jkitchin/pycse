# CLAUDE.md - Project Guide for AI Assistants

This file provides context for Claude and other AI assistants working with this codebase.

## Project Overview

**pycse** (Python Computations in Science and Engineering) is a Python library for scientific computing and engineering calculations. It provides:

- Regression and curve fitting with confidence intervals
- ODE solving with units support
- Machine learning models (sklearn-compatible)
- JAX-based neural networks
- Design of experiments utilities

**Version**: see `version` in `pyproject.toml` (single source of truth; `pycse.__version__` reads the installed metadata)
**Python**: >=3.12
**License**: GPL-3.0-or-later

## Directory Structure

```
pycse/
├── src/pycse/           # Main source code
│   ├── PYCSE.py         # Core regression functions (regress, nlinfit, etc.)
│   ├── cli.py           # `pycse` CLI (launch, mcp, skill)
│   ├── skill/           # Claude Code skill: SKILL.md + reference/*.md
│   ├── beginner.py      # Beginner-friendly utilities
│   ├── utils.py         # General utilities
│   ├── sklearn/         # sklearn-compatible ML models
│   │   ├── kan.py       # Kolmogorov-Arnold Networks
│   │   ├── jax_*.py     # JAX-based neural networks
│   │   ├── dpose.py     # Shallow ensemble with uncertainty
│   │   ├── zenn/        # Zentropy-enhanced neural networks
│   │   ├── tests/       # active_learning and zenn tests (also in testpaths)
│   │   └── ...
│   ├── tests/           # Main test suite
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
- **pycse-tests-slow.yaml** - Slow tests on release tags (`v*`) or manual dispatch (~40 min)

## Code Style

- **Linter**: Ruff (configured in pyproject.toml)
- **Line length**: 100 characters
- **Target**: Python 3.12+

```bash
# Run linter
ruff check src/pycse

# Fix auto-fixable issues
ruff check --fix src/pycse
```

## Key Modules

### Core (`src/pycse/PYCSE.py`)
- `regress()` - Linear regression with confidence intervals
- `polyfit()` / `polyval()` - Polynomial fits with intervals
- `predict()` - Prediction intervals for linear fits
- `nlinfit()` - Nonlinear curve fitting
- `nlpredict()` - Prediction intervals for nonlinear fits
- `ivp()` - ODE initial value problems (`solve_ivp` with defaults)
- Prediction intervals (`yint`) have shape `(n, 2)`: lower, upper

### sklearn Models (`src/pycse/sklearn/`)
All models follow sklearn's estimator API (fit/predict/score). Public names are
lazily exported from `pycse.sklearn` (the `_LAZY` table in `sklearn/__init__.py`).

- `KAN`, `KANLLPR` - Kolmogorov-Arnold Networks
- `JAXMonotonicRegressor` - Monotonic neural networks (JAX)
- `JAXPeriodicRegressor` - Periodic neural networks (JAX)
- `JAXICNNRegressor` - Input-convex neural networks (JAX)
- `DPOSE` - Shallow ensemble with uncertainty
- `KfoldNN` - K-fold neural network ensemble
- `LLPRRegressor`, `NNBR`, `NNGMM`, `LeafModelRegressor`, `LinearRegressionUQ`
- `SISSO`, `SISSOEnsemble` - Symbolic regression (optional `sisso` extra)
- `ZENNRegressor`, `ZENNClassifier` - Zentropy-enhanced neural networks
- `ActiveLearner` - Active learning loop (`active_learning.py`)

### Utilities
- `hashcache.py` - Disk caching for expensive computations
- `beginner.py` - Simplified interfaces for teaching
- `mcp.py` - MCP server (`pycse_mcp`, optional `mcp` extra)
- `sklearn/mip.py` - MIP export of piecewise-linear models to Pyomo (optional `mip` extra).
  Models implement `mip_network(**options)` (a `MIPNetwork`, `TreeModel` or `SymbolicModel`)
  and get `to_pyomo()` from `MIPExportMixin`; tests in `test_sklearn_mip.py` solve with HiGHS
  and compare to `predict()` (std objectives and SISSO tests need gurobipy and skip without it).
  Walkthrough: `pycse-jb/.../docs/mip.ipynb`

### Claude Code skill (`src/pycse/skill/`)
`pycse skill install` copies this directory to `~/.claude/skills/pycse/`. Keep the
examples in it in sync with the API when signatures change; `test_cli.py` checks
the frontmatter and that its links resolve.

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
- Include tests in `src/pycse/tests/` (`test_sklearn_<model>.py`); both test
  directories are in `testpaths` in pyproject.toml

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
