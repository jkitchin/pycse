"""Tests for supyrvisor module."""

import pytest
from pycse.supyrvisor import (
    supervisor,
    manager,
    check_result,
    check_exception,
    TooManyErrorsException,
)


# Tests for supervisor decorator
class TestSupervisor:
    """Tests for the supervisor decorator."""

    def test_basic_success_no_checks(self):
        """Test function that succeeds with no checks."""

        @supervisor()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_with_check_function_no_fixes_needed(self):
        """Test with check function that passes."""

        def check_positive(args, kwargs, result):
            if result > 0:
                return None  # all good
            return (args, kwargs)  # needs to be rerun

        @supervisor(check_funcs=(check_positive,))
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_with_check_function_fixes_needed(self):
        """Test with check function that fixes the result."""
        call_count = {"count": 0}

        def check_positive(args, kwargs, result):
            if result > 0:
                return None
            # Make second argument positive
            return (args[0], abs(args[1])), kwargs

        @supervisor(check_funcs=(check_positive,))
        def add(a, b):
            call_count["count"] += 1
            return a + b

        result = add(2, -3)
        assert result == 5  # After fix: 2 + 3
        assert call_count["count"] == 2  # Called twice

    def test_multiple_check_functions(self):
        """Test with multiple check functions."""

        def check_positive(args, kwargs, result):
            if result > 0:
                return None
            return (abs(args[0]), abs(args[1])), kwargs

        def check_even(args, kwargs, result):
            if result % 2 == 0:
                return None
            return (args[0], args[1] + 1), kwargs

        @supervisor(check_funcs=(check_positive, check_even))
        def add(a, b):
            return a + b

        # First run: 2 + 3 = 5 (odd, fails check_even)
        # Second run: 2 + 4 = 6 (even, passes both checks)
        result = add(2, 3)
        assert result == 6

    def test_exception_handling_no_fixer(self):
        """Test exception raised when no exception_funcs defined."""

        @supervisor()
        def divide(a, b):
            return a / b

        with pytest.raises(ZeroDivisionError):
            divide(10, 0)

    def test_exception_handling_with_fixer(self):
        """Test exception handler that fixes the issue."""
        call_count = {"count": 0}

        def fix_zero_division(args, kwargs, exc):
            if isinstance(exc, ZeroDivisionError):
                # Change divisor to 1
                return (args[0], 1), kwargs
            return None

        @supervisor(exception_funcs=(fix_zero_division,))
        def divide(a, b):
            call_count["count"] += 1
            return a / b

        result = divide(10, 0)
        assert result == 10.0  # After fix: 10 / 1
        assert call_count["count"] == 2

    def test_exception_handler_cannot_fix(self):
        """Test exception handler that cannot fix the issue."""

        def fix_zero_division(args, kwargs, exc):
            if isinstance(exc, ZeroDivisionError):
                return (args[0], 1), kwargs
            return None  # Can't fix other exceptions

        @supervisor(exception_funcs=(fix_zero_division,))
        def divide(a, b):
            if b == 1:
                raise ValueError("b cannot be 1")
            return a / b

        with pytest.raises(ValueError, match="b cannot be 1"):
            divide(10, 0)  # Will fix to b=1, then raise ValueError

    def test_max_errors_reached(self):
        """Test that TooManyErrorsException is raised when max_errors reached."""

        def always_fail_check(args, kwargs, result):
            # Always return new args, forcing continuous rerun
            return (args[0] + 1, args[1]), kwargs

        @supervisor(check_funcs=(always_fail_check,), max_errors=3)
        def add(a, b):
            return a + b

        with pytest.raises(TooManyErrorsException, match="Maximum number of errors"):
            add(1, 2)

    def test_verbose_mode(self, capsys):
        """Test verbose mode prints fix messages."""

        def check_positive(args, kwargs, result):
            if result > 0:
                return None
            return (abs(args[0]), abs(args[1])), kwargs

        @supervisor(check_funcs=(check_positive,), verbose=True)
        def add(a, b):
            return a + b

        add(-2, -3)
        captured = capsys.readouterr()
        assert "Proposed fix in check_positive" in captured.out

    def test_kwargs_handling(self):
        """Test that kwargs are properly handled."""

        def check_positive(args, kwargs, result):
            if result > 0:
                return None
            return args, {"a": abs(kwargs["a"]), "b": abs(kwargs["b"])}

        @supervisor(check_funcs=(check_positive,))
        def add(a, b):
            return a + b

        result = add(a=-2, b=-3)
        assert result == 5


# Tests for manager decorator
class TestManager:
    """Tests for the manager decorator."""

    def test_basic_success_no_checkers(self):
        """Test function that succeeds with no checkers."""

        @manager()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_with_result_checker(self):
        """Test with result checker."""

        @check_result
        def check_positive(arguments, result):
            if result > 0:
                return None
            return {"a": abs(arguments["a"]), "b": abs(arguments["b"])}

        @manager(checkers=(check_positive,))
        def add(a, b):
            return a + b

        result = add(-2, -3)
        assert result == 5

    def test_with_exception_checker(self):
        """Test with exception checker."""

        @check_exception
        def fix_zero_division(arguments, exc):
            if isinstance(exc, ZeroDivisionError):
                return {"a": arguments["a"], "b": 1}
            return None

        @manager(checkers=(fix_zero_division,))
        def divide(a, b):
            return a / b

        result = divide(10, 0)
        assert result == 10.0

    def test_mixed_checkers(self):
        """Test with both result and exception checkers."""

        @check_result
        def check_positive(arguments, result):
            if result > 0:
                return None
            return {"a": abs(arguments["a"]), "b": abs(arguments["b"])}

        @check_exception
        def fix_zero_division(arguments, exc):
            if isinstance(exc, ZeroDivisionError):
                return {"a": arguments["a"], "b": 1}
            return None

        @manager(checkers=(check_positive, fix_zero_division))
        def safe_add(a, b):
            if b == 0:
                raise ZeroDivisionError("b is zero")
            return a + b

        # Test exception path
        result = safe_add(10, 0)
        assert result == 11

        # Test check path
        result = safe_add(-5, -3)
        assert result == 8

    def test_default_arguments(self):
        """Test that default arguments are properly included."""

        @check_result
        def check_value(arguments, result):
            if result == 15:  # 10 + 5 (default)
                return None
            return {"a": 10, "b": 5}

        @manager(checkers=(check_value,))
        def add(a, b=5):
            return a + b

        result = add(3)  # 3 + 5 = 8, should be fixed to 10 + 5 = 15
        assert result == 15

    def test_max_errors_manager(self):
        """Test max_errors with manager decorator."""

        @check_result
        def always_fail(arguments, result):
            return {"a": arguments["a"] + 1, "b": arguments["b"]}

        @manager(checkers=(always_fail,), max_errors=3)
        def add(a, b):
            return a + b

        with pytest.raises(TooManyErrorsException, match="Maximum number of errors"):
            add(1, 2)

    def test_verbose_mode_manager(self, capsys):
        """Test verbose mode with manager."""

        @check_result
        def check_positive(arguments, result):
            if result > 0:
                return None
            return {"a": abs(arguments["a"]), "b": abs(arguments["b"])}

        @manager(checkers=(check_positive,), verbose=True)
        def add(a, b):
            return a + b

        add(-2, -3)
        captured = capsys.readouterr()
        assert "Proposed fix" in captured.out


# Tests for check_result and check_exception decorators
class TestCheckDecorators:
    """Tests for check_result and check_exception decorators."""

    def test_check_result_with_exception(self):
        """Test that check_result returns None for exceptions."""

        @check_result
        def checker(arguments, result):
            return {"new": "args"}

        # Should return None when result is an Exception
        assert checker({"a": 1}, ValueError("error")) is None

    def test_check_result_with_normal_result(self):
        """Test that check_result works with normal results."""

        @check_result
        def checker(arguments, result):
            if result > 0:
                return None
            return {"a": 1}

        # Should call the function normally
        assert checker({"a": -1}, -5) == {"a": 1}
        assert checker({"a": 1}, 5) is None

    def test_check_exception_with_exception(self):
        """Test that check_exception handles exceptions."""

        @check_exception
        def fixer(arguments, exc):
            if isinstance(exc, ValueError):
                return {"fixed": True}
            return None

        # Should handle exception
        assert fixer({"a": 1}, ValueError("error")) == {"fixed": True}
        assert fixer({"a": 1}, TypeError("error")) is None

    def test_check_exception_with_normal_result(self):
        """Test that check_exception returns None for normal results."""

        @check_exception
        def fixer(arguments, exc):
            return {"fixed": True}

        # Should return None when result is not an exception
        assert fixer({"a": 1}, 42) is None

    def test_check_result_with_callable_class(self):
        """Test check_result with callable class."""

        class ResultChecker:
            def __call__(self, arguments, result):
                return None if result > 0 else {"a": 1}

        checker = check_result(ResultChecker())
        assert checker({"a": -1}, -5) == {"a": 1}

    def test_check_exception_with_callable_class(self):
        """Test check_exception with callable class."""

        class ExceptionFixer:
            def __call__(self, arguments, exc):
                if isinstance(exc, ValueError):
                    return {"fixed": True}
                return None

        fixer = check_exception(ExceptionFixer())
        assert fixer({"a": 1}, ValueError("error")) == {"fixed": True}


# Integration tests
class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_retry_http_request_simulation(self):
        """Simulate retrying an HTTP request with backoff."""
        call_log = []

        def fix_timeout(args, kwargs, exc):
            if isinstance(exc, TimeoutError):
                call_log.append("retry")
                # Simulate retry logic - double the timeout
                new_timeout = kwargs.get("timeout", 1) * 2
                return args, {"timeout": new_timeout}
            return None

        @supervisor(exception_funcs=(fix_timeout,), max_errors=3)
        def fetch(url, timeout=1):
            call_log.append(f"fetch: timeout={timeout}")
            if timeout < 4:
                raise TimeoutError("Request timed out")
            return f"Data from {url}"

        result = fetch("http://example.com", timeout=1)
        assert result == "Data from http://example.com"
        assert len([x for x in call_log if x == "retry"]) == 2

    def test_file_processing_with_validation(self):
        """Simulate processing a file with validation."""

        @check_result
        def validate_output(arguments, result):
            if len(result) > 0:
                return None
            # If empty, try processing again with different encoding
            return {"data": arguments["data"], "encoding": "latin-1"}

        @manager(checkers=(validate_output,))
        def process_file(data, encoding="utf-8"):
            if encoding == "utf-8" and "special" in data:
                return ""  # Empty result, needs retry
            return f"Processed: {data}"

        result = process_file("special data")
        assert result == "Processed: special data"

    def test_numerical_solver_convergence(self):
        """Simulate numerical solver that adjusts tolerance."""
        iterations = {"count": 0}

        def check_converged(args, kwargs, result):
            if abs(result - 3.14159) < kwargs.get("tolerance", 0.01):
                return None  # Converged
            # Reduce tolerance and try again
            new_tol = kwargs.get("tolerance", 0.01) / 10
            return args, {**kwargs, "tolerance": new_tol}

        @supervisor(check_funcs=(check_converged,), max_errors=3)
        def solve(tolerance=0.1):
            iterations["count"] += 1
            if tolerance < 0.001:
                return 3.14159  # "Converged" solution
            return 3.0  # Not converged yet

        result = solve()
        assert abs(result - 3.14159) < 0.01
        assert iterations["count"] > 1
