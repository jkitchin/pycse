"""Tests for the beginner module."""

from pycse import feq
from pycse.beginner import (
    first,
    second,
    third,
    fourth,
    fifth,
    rest,
    last,
    butlast,
    nth,
    cut,
    integrate,
    nsolve,
    NsolveError,
    IntegrateError,
)
import numpy as np
import pytest

a = [1, 2, 3, 4, 5]


def test_first_a():
    """Check first."""
    assert first(a) == 1


def test_first_b():
    """Check first raises exception."""
    with pytest.raises(Exception):
        assert first(1) == 1


def test_second_a():
    """Test second."""
    assert second(a) == 2


def test_second_b():
    """Check second raises exception on integer."""
    with pytest.raises(Exception):
        assert second(1) == 0


def test_second_c():
    """Check second raises exception on too short list."""
    with pytest.raises(Exception):
        assert second([1]) == 0


def test_third_a():
    """Test third."""
    assert third(a) == 3


def test_third_b():
    """Check third raises exception on integer."""
    with pytest.raises(Exception):
        assert third(1) == 0


def test_third_c():
    """Check third raises exception on short list."""
    with pytest.raises(Exception):
        assert third([1, 2]) == 0


def test_fourth_a():
    """Test fourth."""
    assert fourth(a) == 4


def test_fourth_b():
    """Check fourth raises exception on integer."""
    with pytest.raises(Exception):
        assert fourth(1) == 0


def test_fourth_c():
    """Check fourth raises exception on short list."""
    with pytest.raises(Exception):
        assert fourth([1]) == 0


def test_fifth_a():
    "Check fifth element."
    assert fifth(a) == 5


def test_fifth_b():
    """Check fourth raises exception on integer."""
    with pytest.raises(Exception):
        assert fifth(1) == 0


def test_fifth_c():
    """Check fifth raises exception on short list."""
    with pytest.raises(Exception):
        assert fifth([1]) == 0


def test_rest():
    """Test rest."""
    assert rest(a) == [2, 3, 4, 5]


def test_rest_2():
    """Check rest raises exception."""
    with pytest.raises(Exception):
        assert rest(1) == [2, 3, 4, 5]


def test_last():
    """Test last."""
    assert last(a) == 5


def test_last_b():
    with pytest.raises(Exception):
        assert last(5)


def test_butlast():
    assert butlast(a) == [1, 2, 3, 4]


def test_nth():
    assert nth(a, 1) == 2


def test_nth_a():
    with pytest.raises(Exception):
        assert nth(a, 6) is None


def test_nth_a_exc():
    with pytest.raises(Exception):
        assert nth(6) is None


def test_cut_a():
    assert cut(a, 1, None, 2) == [2, 4]


def test_cut_a_exc4():
    with pytest.raises(Exception):
        assert cut(1, 1, None, 2) == [2, 4]


def test_integrate():
    assert integrate(lambda x: 1, 0, 1) == 1


def test_nsolve():
    assert feq(nsolve(lambda x: x - 3, 2), 3)


def test_nsolve_2():
    with pytest.raises(Exception):
        assert feq(nsolve(lambda x: x + 3, 2), 3)


def test_nsolve_3():
    assert (nsolve(lambda X: [X[0] - 3, X[1] - 4], [2, 2]) == [3, 4]).all()


def test_nsolve_error_class():
    """nsolve raises a specific exception that is still an Exception."""
    with pytest.raises(NsolveError):
        nsolve(lambda x: x**2 + 1, 2)
    assert issubclass(NsolveError, RuntimeError)


def test_nsolve_extra_args():
    """Extra positional args go to the objective, not fsolve's fprime slot."""
    assert feq(nsolve(lambda x, a, b: a * x - b, 1.0, 2.0, 6.0), 3.0)


def test_integrate_tolerance_kwarg():
    """tolerance is consumed by integrate and not passed on to quad."""
    assert feq(integrate(lambda x: x, 0, 1, tolerance=1e-3), 0.5)


def test_integrate_large_magnitude():
    """Correct large integrals should not fail an absolute-only error check."""
    val = integrate(lambda x: 1e8 * np.exp(-(x**2)), 0, 5)
    assert np.isclose(val, 1e8 * np.sqrt(np.pi) / 2)


def test_integrate_extra_args():
    """Extra positional args are passed to f."""
    assert feq(integrate(lambda x, a, b: a * x + b, 0, 1, 3.0, 1.0), 2.5)


def test_integrate_error_no_message():
    """A too-large error without a quad message gives a helpful error."""
    with pytest.raises(IntegrateError, match="integral error"):
        integrate(lambda x: np.exp(x), 0, 1, tolerance=1e-30, rtol=0)


def test_integrate_error_with_message():
    """The quad warning message is included when available."""
    with pytest.raises(IntegrateError, match="subdivisions"):
        integrate(lambda x: 1 / np.sqrt(abs(x - 0.3)), 0, 1, limit=2)
