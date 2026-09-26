"""Tests for utils module."""

import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch
from pycse.utils import feq, fgt, flt, fle, fge, ignore_exception, read_gsheet


def test_feq():
    """Test fuzzy equal."""
    assert feq(1, 1)
    assert not (feq(1, 0))


def test_fgt():
    """Test fuzzy greater than."""
    assert fgt(2, 1)
    assert not (fgt(2, 4))
    assert not (fgt(2, 2))


def test_flt():
    """Test fuzzy less than."""
    assert flt(1, 2)
    assert not (flt(2, 1))
    assert not (flt(1, 1))


def test_fle():
    """Test fuzzy less than or equal to."""
    assert fle(1, 2)
    assert fle(1, 1)
    assert not (fle(2, 1))


def test_fge():
    """Test fuzzy greater than or equal to."""
    assert fge(2, 1)
    assert fge(2, 2)
    assert not (fge(1, 2))


def test_ie():
    """Test ignore exception."""
    with ignore_exception(ZeroDivisionError):
        print(1 / 0)
    assert True


def test_ie_no_done_print(capsys):
    """ignore_exception should not print unconditionally."""
    with ignore_exception(ZeroDivisionError):
        pass
    assert capsys.readouterr().out == ""

    with ignore_exception(ZeroDivisionError):
        1 / 0
    assert capsys.readouterr().out == "caught division by zero\n"


def test_ie_other_exceptions_propagate():
    """Exceptions not listed are not ignored."""
    with pytest.raises(ValueError):
        with ignore_exception(ZeroDivisionError):
            raise ValueError("no")


def test_fuzzy_scalar_returns_bool():
    """Scalar comparisons return Python bools."""
    for f in (feq, flt, fgt, fle, fge):
        assert type(f(1.0, 2.0)) is bool
        assert type(f(np.float64(1.0), np.float64(1.0))) is bool


def test_fuzzy_arrays():
    """Fuzzy comparisons work elementwise on arrays."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0])
    np.testing.assert_array_equal(feq(x, y), [True, False, False])
    np.testing.assert_array_equal(flt(x, y), [False, True, False])
    np.testing.assert_array_equal(fgt(x, y), [False, False, True])
    np.testing.assert_array_equal(fle(x, y), [True, True, False])
    np.testing.assert_array_equal(fge(x, y), [True, False, True])


def test_fuzzy_rtol():
    """Relative tolerance allows comparing large numbers."""
    a, b = 1e6, 1e6 * (1 + 1e-15)
    # default absolute tolerance is unchanged
    assert not feq(a, b)
    assert feq(a, b, rtol=1e-12)
    assert fle(b, a, rtol=1e-12)
    assert fge(a, b, rtol=1e-12)
    assert not flt(a, b, rtol=1e-12)
    assert not fgt(b, a, rtol=1e-12)
    assert flt(1e6, 2e6, rtol=1e-12)


# Tests for read_gsheet
def test_read_gsheet_invalid_url():
    """Test that read_gsheet raises exception for invalid URL."""
    with pytest.raises(Exception, match="does not seem to be for a sheet"):
        read_gsheet("https://example.com/not-a-sheet")


@patch("pycse.utils.pd.read_csv")
def test_read_gsheet_with_gid(mock_read_csv):
    """Test read_gsheet with a Google Sheet URL containing gid."""
    # Mock the read_csv to avoid network calls
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = mock_df

    url = "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=456"
    result = read_gsheet(url)

    # Verify the correct export URL was constructed
    expected_url = "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=456"
    mock_read_csv.assert_called_once_with(expected_url)

    # Verify the dataframe is returned
    pd.testing.assert_frame_equal(result, mock_df)


@patch("pycse.utils.pd.read_csv")
def test_read_gsheet_without_gid(mock_read_csv):
    """Test read_gsheet with URL without gid (should default to 0)."""
    mock_df = pd.DataFrame({"col1": [1, 2]})
    mock_read_csv.return_value = mock_df

    url = "https://docs.google.com/spreadsheets/d/ABC123/edit"
    result = read_gsheet(url)

    # Should default to gid=0
    expected_url = "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=0"
    mock_read_csv.assert_called_once_with(expected_url)

    pd.testing.assert_frame_equal(result, mock_df)


@patch("pycse.utils.pd.read_csv")
def test_read_gsheet_with_kwargs(mock_read_csv):
    """Test that args and kwargs are passed through to pd.read_csv."""
    mock_df = pd.DataFrame({"col1": [1]})
    mock_read_csv.return_value = mock_df

    url = "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=0"
    read_gsheet(url, header=0, skiprows=2)

    expected_url = "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=0"
    mock_read_csv.assert_called_once_with(expected_url, header=0, skiprows=2)
