"""Tests for pyroxy module."""

import pytest
import numpy as np
import os
from unittest.mock import patch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from pycse.pyroxy import _Surrogate, Surrogate, MaxCallsExceededException


@pytest.fixture
def simple_model():
    """Create a simple GPR model for testing."""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)


@pytest.fixture
def simple_function():
    """A simple test function."""

    def f(X):
        return np.sin(X).flatten()

    return f


class TestSurrogateClass:
    """Tests for the _Surrogate class."""

    def test_initialization(self, simple_function, simple_model):
        """Test basic initialization."""
        surr = _Surrogate(simple_function, simple_model, tol=0.5, max_calls=100)

        assert surr.func == simple_function
        assert surr.model == simple_model
        assert surr.tol == 0.5
        assert surr.max_calls == 100
        assert surr.verbose is False
        assert surr.xtrain is None
        assert surr.ytrain is None
        assert surr.ntrain == 0
        assert surr.surrogate == 0
        assert surr.func_calls == 0

    def test_add_method(self, simple_function, simple_model):
        """Test the add method."""
        surr = _Surrogate(simple_function, simple_model)

        X = np.array([[0.0], [1.0], [2.0]])
        y = surr.add(X)

        assert surr.func_calls == 1
        assert surr.ntrain == 1
        assert np.array_equal(surr.xtrain, X)
        assert np.array_equal(surr.ytrain, y)
        assert np.allclose(y, np.sin(X).flatten())

    def test_add_multiple_times(self, simple_function, simple_model):
        """Test adding data multiple times."""
        surr = _Surrogate(simple_function, simple_model)

        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[2.0], [3.0]])

        surr.add(X1)
        surr.add(X2)

        assert surr.func_calls == 2
        assert surr.ntrain == 2
        assert len(surr.xtrain) == 4
        assert len(surr.ytrain) == 4

    def test_call_initializes_model(self, simple_function, simple_model):
        """Test that calling surrogate initializes the model."""
        surr = _Surrogate(simple_function, simple_model, tol=0.1)

        X = np.array([[0.5]])
        y = surr(X)

        assert surr.func_calls == 1
        assert surr.ntrain == 1
        assert surr.xtrain is not None
        assert np.allclose(y, np.sin(X).flatten())

    def test_call_uses_surrogate_when_accurate(self, simple_function, simple_model):
        """Test that surrogate is used when predictions are accurate enough."""
        surr = _Surrogate(simple_function, simple_model, tol=10.0)  # High tolerance

        # Initialize with some data
        X_train = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        surr.add(X_train)

        initial_func_calls = surr.func_calls

        # Call at a point where surrogate should be accurate
        X_test = np.array([[np.pi]])
        _ = surr(X_test)

        # Should not have called the function again
        assert surr.func_calls == initial_func_calls
        assert surr.surrogate == 1

    def test_call_retrains_when_inaccurate(self, simple_function, simple_model):
        """Test that model retrains when predictions are inaccurate."""
        surr = _Surrogate(simple_function, simple_model, tol=0.001)  # Low tolerance

        # Initialize with limited data
        X_train = np.array([[0.0], [1.0]])
        surr.add(X_train)

        initial_func_calls = surr.func_calls

        # Call at a point far from training data
        X_test = np.array([[5.0]])
        _ = surr(X_test)

        # Should have called the function and retrained
        assert surr.func_calls > initial_func_calls
        assert surr.ntrain == 2

    def test_max_calls_exceeded_in_add(self, simple_function, simple_model):
        """Test that max_calls limit is enforced in add method."""
        surr = _Surrogate(simple_function, simple_model, max_calls=2)

        X = np.array([[0.0]])
        surr.add(X)
        surr.add(X)

        with pytest.raises(
            MaxCallsExceededException, match="Max func calls \\(2\\) will be exceeded"
        ):
            surr.add(X)

    def test_max_calls_exceeded_in_call(self, simple_function, simple_model):
        """Test that max_calls limit is enforced in __call__."""
        surr = _Surrogate(simple_function, simple_model, max_calls=1, tol=0.001)

        X = np.array([[0.0]])
        surr(X)  # First call initializes

        # Second call at a different point should raise exception
        # Use a far point to ensure surrogate error is large
        X2 = np.array([[10.0]])
        with pytest.raises(
            MaxCallsExceededException, match="Max func calls \\(1\\) will be exceeded"
        ):
            surr(X2)

    def test_verbose_mode(self, simple_function, simple_model, capsys):
        """Test verbose output."""
        surr = _Surrogate(simple_function, simple_model, verbose=True, tol=0.001)

        # Initialize
        X = np.array([[0.0]])
        surr(X)

        captured = capsys.readouterr()
        # Check for initialization message (case-insensitive)
        assert "running" in captured.out.lower()

        # Test verbose output when retraining
        X2 = np.array([[5.0]])
        surr(X2)

        captured = capsys.readouterr()
        assert "greater than" in captured.out

    def test_test_method(self, simple_function, simple_model):
        """Test the test method."""
        surr = _Surrogate(simple_function, simple_model, tol=0.1)

        # Train with some data
        X_train = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        surr.add(X_train)

        # Test at a nearby point
        X_test = np.array([[np.pi / 2]])
        result = surr.test(X_test)

        # Result should be boolean
        assert isinstance(result, (bool, np.bool_))
        assert surr.func_calls == 2  # 1 from add, 1 from test

    def test_test_method_verbose(self, simple_function, simple_model, capsys):
        """Test verbose output in test method."""
        surr = _Surrogate(simple_function, simple_model, tol=0.1, verbose=True)

        X_train = np.array([[0.0]])
        surr.add(X_train)

        X_test = np.array([[1.0]])
        surr.test(X_test)

        captured = capsys.readouterr()
        assert "Testing" in captured.out

    def test_test_method_max_calls(self, simple_function, simple_model):
        """Test that test method respects max_calls."""
        surr = _Surrogate(simple_function, simple_model, max_calls=1)

        X = np.array([[0.0]])
        surr.add(X)

        with pytest.raises(
            MaxCallsExceededException, match="Max func calls \\(1\\) will be exceeded"
        ):
            surr.test(X)

    def test_str_representation(self, simple_function, simple_model):
        """Test string representation."""
        surr = _Surrogate(simple_function, simple_model)

        X = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        surr.add(X)

        string_repr = str(surr)

        assert "data points" in string_repr
        assert "fitted" in string_repr
        assert "model score" in string_repr
        assert "MAE" in string_repr
        assert "RMSE" in string_repr

    def test_plot_method(self, simple_function, simple_model):
        """Test plot method returns expected output."""
        surr = _Surrogate(simple_function, simple_model)

        X = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        surr.add(X)

        with patch("matplotlib.pyplot.plot") as mock_plot:
            with patch("matplotlib.pyplot.fill_between"):
                with patch("matplotlib.pyplot.xlabel"):
                    with patch("matplotlib.pyplot.ylabel"):
                        with patch("matplotlib.pyplot.title"):
                            _ = surr.plot()

        # plot should have been called
        mock_plot.assert_called_once()

    def test_dump_and_load(self, simple_function, simple_model, tmp_path):
        """Test dump and load functionality."""
        surr = _Surrogate(simple_function, simple_model)

        X = np.array([[0.0], [1.0], [2.0]])
        surr.add(X)

        # Dump to temporary file
        fname = tmp_path / "test_surrogate.pkl"
        returned_fname = surr.dump(str(fname))

        assert os.path.exists(fname)
        assert returned_fname == str(fname)

        # Load and verify
        loaded_surr = Surrogate.load(str(fname))

        assert loaded_surr.func_calls == surr.func_calls
        assert loaded_surr.ntrain == surr.ntrain
        assert np.array_equal(loaded_surr.xtrain, surr.xtrain)
        assert np.array_equal(loaded_surr.ytrain, surr.ytrain)


class TestSurrogateDecorator:
    """Tests for the Surrogate decorator function."""

    def test_decorator_basic(self, simple_model):
        """Test basic decorator usage."""

        @Surrogate(model=simple_model, tol=0.5)
        def my_func(X):
            return np.sin(X).flatten()

        assert isinstance(my_func, _Surrogate)
        assert my_func.tol == 0.5

    def test_decorator_with_max_calls(self, simple_model):
        """Test decorator with max_calls."""

        @Surrogate(model=simple_model, max_calls=10, verbose=True)
        def my_func(X):
            return X**2

        assert my_func.max_calls == 10
        assert my_func.verbose is True

    def test_decorated_function_works(self, simple_model):
        """Test that decorated function actually works."""

        @Surrogate(model=simple_model, tol=1.0)
        def my_func(X):
            return (X**2).flatten()

        X = np.array([[1.0], [2.0], [3.0]])
        y = my_func(X)

        assert my_func.func_calls == 1
        assert len(y) == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_max_calls(self, simple_function, simple_model):
        """Test with max_calls=0."""
        surr = _Surrogate(simple_function, simple_model, max_calls=0)

        # Should raise immediately on first call
        with pytest.raises(MaxCallsExceededException, match="Max func calls"):
            # Use add() which always calls the function
            surr.add(np.array([[0.0]]))

    def test_negative_max_calls_means_unlimited(self, simple_function, simple_model):
        """Test that negative max_calls means no limit."""
        surr = _Surrogate(simple_function, simple_model, max_calls=-1)

        # Should be able to call many times
        for i in range(10):
            surr.add(np.array([[float(i)]]))

        assert surr.func_calls == 10  # No exception raised

    def test_multivariate_function(self, simple_model):
        """Test with multivariate function."""

        def multi_func(X):
            return np.sum(X, axis=1)

        surr = _Surrogate(multi_func, simple_model)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = surr.add(X)

        assert len(y) == 2
        assert surr.func_calls == 1
