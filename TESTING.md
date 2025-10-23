# Testing Guide for pycse

This document provides guidelines for writing and maintaining tests in the pycse project.

## Quick Start

### Running Tests Locally

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/pycse --cov-report=html --cov-report=term-missing

# Run specific test file
pytest src/pycse/tests/test_pycse.py

# Run tests matching a pattern
pytest -k "test_regress"

# View HTML coverage report
open htmlcov/index.html
```

### Running Tests in CI

Tests run automatically on every push and pull request via GitHub Actions. Coverage reports are uploaded to Codecov.

## Test Structure

### Test Organization

```
src/pycse/tests/
├── conftest.py           # Shared fixtures and utilities
├── test_pycse.py         # Tests for PYCSE.py (core regression)
├── test_beginner.py      # Tests for beginner.py
├── test_utils.py         # Tests for utils.py
└── test_sklearn_*.py     # Tests for sklearn modules (future)
```

### Test File Naming

- Test files must start with `test_`
- Test functions must start with `test_`
- Test classes must start with `Test`

### Example Test Structure

```python
"""Tests for module_name.py."""

import numpy as np
import pytest
from pycse.module_name import function_to_test


def test_basic_functionality():
    """Test basic happy path."""
    result = function_to_test([1, 2, 3])
    assert result == expected_value


def test_edge_case():
    """Test edge case behavior."""
    result = function_to_test([])
    assert result is None


def test_error_handling():
    """Test that appropriate errors are raised."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)


@pytest.mark.slow
def test_expensive_computation():
    """Test that takes >1 second."""
    # Use @pytest.mark.slow for tests that are expensive
    pass
```

## Testing Guidelines

### 1. Test Coverage Requirements

- **New code**: Minimum 80% coverage required
- **Existing code**: Incremental improvements, no regressions
- **Critical paths**: 100% coverage (marked with `@pytest.mark.regression`)
- **Overall project**: Target 70%+ coverage

### 2. What to Test

#### Happy Path (Required)
```python
def test_linear_regression_basic():
    """Test linear regression on perfect data."""
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 3, 5, 7])  # y = 2x + 1

    b, bint, se = regress(np.column_stack([x, x**0]), y)

    assert np.isclose(b[0], 2.0)  # slope
    assert np.isclose(b[1], 1.0)  # intercept
```

#### Edge Cases (Required)
```python
def test_linear_regression_single_point():
    """Test behavior with minimal data."""
    # Test with edge cases like single points, collinear data, etc.
    pass
```

#### Error Conditions (Required)
```python
def test_linear_regression_invalid_shape():
    """Test that proper errors are raised for invalid input."""
    with pytest.raises(ValueError):
        regress(np.array([1, 2]), np.array([1, 2, 3]))  # Mismatched shapes
```

#### Numerical Accuracy (For Scientific Code)
```python
def test_numerical_stability():
    """Test numerical accuracy with known solutions."""
    # Use problems with analytical solutions
    # Compare against reference implementations
    # Check condition numbers for ill-conditioned problems
    pass
```

### 3. Using Fixtures

```python
def test_with_fixture(linear_data):
    """Use fixtures from conftest.py."""
    x, y = linear_data
    result = function_to_test(x, y)
    assert result is not None
```

Common fixtures available in `conftest.py`:
- `linear_data`: Simple linear relationship
- `linear_data_with_noise`: Linear with noise
- `quadratic_data`: Quadratic relationship
- `exponential_data`: Exponential relationship
- `simple_ode_solution`: ODE solution

### 4. Testing Numerical Code

```python
def test_numerical_accuracy():
    """Test numerical functions with appropriate tolerances."""
    from pycse.tests.conftest import assert_allclose_with_tol

    actual = compute_something()
    expected = known_solution()

    # Use appropriate tolerances for floating point comparisons
    assert_allclose_with_tol(actual, expected, rtol=1e-5, atol=1e-8)
```

### 5. Testing Machine Learning Models (sklearn)

```python
@pytest.mark.sklearn
def test_sklearn_api_compatibility():
    """Test that model follows sklearn API."""
    from sklearn.utils.estimator_checks import check_estimator
    from pycse.sklearn.my_model import MyModel

    # This runs sklearn's standard test suite
    check_estimator(MyModel())


def test_fit_predict():
    """Test basic fit/predict cycle."""
    model = MyModel()
    X_train, y_train = get_training_data()
    X_test = get_test_data()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test),)
```

### 6. Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

def test_with_mocked_api():
    """Test code that calls external APIs."""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'data': 'value'}
        )

        result = function_that_calls_api()
        assert result == expected_value
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow  # Tests that take >1 second
@pytest.mark.integration  # Integration tests
@pytest.mark.regression  # Critical path tests
@pytest.mark.sklearn  # sklearn module tests
```

Run specific markers:
```bash
# Skip slow tests
pytest -m "not slow"

# Run only regression tests
pytest -m regression
```

## Coverage Best Practices

### 1. Check Coverage for Specific Files

```bash
# Check coverage for a specific module
pytest --cov=src/pycse/PYCSE --cov-report=term-missing

# Check coverage delta (what changed)
pytest --cov=src/pycse --cov-report=html --cov-report=term-missing
```

### 2. Exclude Code from Coverage

Use `# pragma: no cover` for code that doesn't need testing:

```python
def debug_function():  # pragma: no cover
    """Only used for debugging."""
    print("Debug output")

if __name__ == "__main__":  # pragma: no cover
    main()
```

### 3. Coverage Configuration

Coverage settings are in `pyproject.toml`:
- Excludes: tests, examples, __init__.py
- Minimum precision: 2 decimal places
- HTML reports: `htmlcov/` directory

## Continuous Integration

### GitHub Actions Workflow

Tests run on:
- Every push
- Every pull request
- Python 3.12 and 3.13
- Ubuntu Linux

### Coverage Requirements

- Coverage reports uploaded to Codecov
- PR comments show coverage changes
- Coverage must not decrease (enforced in CI)

## Common Testing Patterns

### Testing Statistical Functions

```python
def test_confidence_intervals():
    """Test that confidence intervals are valid."""
    from pycse.tests.conftest import assert_confidence_interval_valid

    b, bint, se = regress(X, y)

    for i, (param, ci) in enumerate(zip(b, bint)):
        assert_confidence_interval_valid(ci, param)
```

### Testing Array Outputs

```python
def test_array_shape():
    """Test that outputs have correct shapes."""
    from pycse.tests.conftest import assert_shape

    result = function_returning_array()
    assert_shape(result, (10, 3))
```

### Testing Error Messages

```python
def test_informative_error():
    """Test that errors provide helpful messages."""
    with pytest.raises(ValueError, match="must be positive"):
        function_with_validation(-1)
```

## Debugging Failed Tests

### 1. Run with Verbose Output

```bash
pytest -vv  # Very verbose
pytest -vv -s  # Also show print statements
```

### 2. Run Specific Test

```bash
pytest src/pycse/tests/test_pycse.py::test_regress -vv
```

### 3. Use pytest Debugger

```bash
pytest --pdb  # Drop into debugger on failure
```

### 4. Check Coverage Report

```bash
pytest --cov=src/pycse --cov-report=html
open htmlcov/index.html
```

## Contributing Tests

When adding new tests:

1. **Write tests for new features** before or alongside implementation
2. **Add tests when fixing bugs** to prevent regression
3. **Aim for 80%+ coverage** on new code
4. **Use descriptive test names** that explain what is being tested
5. **Include docstrings** explaining the test purpose
6. **Keep tests focused** - one concept per test
7. **Use fixtures** from conftest.py when appropriate
8. **Mark expensive tests** with `@pytest.mark.slow`

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [sklearn testing utilities](https://scikit-learn.org/stable/developers/develop.html#testing)

## Questions?

If you have questions about testing:
1. Check existing tests for examples
2. Review this guide
3. Ask in a GitHub issue or PR
