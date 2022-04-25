from pycse.utils import feq, fgt, flt, fle, fge, ignore_exception


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
