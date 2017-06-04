from pycse.utils import *

def test_feq():
    assert feq(1, 1)
    assert not(feq(1, 0))

def test_fgt():
    assert fgt(2, 1)
    assert not(fgt(2, 4))
    assert not(fgt(2, 2))

def test_flt():
    assert flt(1, 2)
    assert not(flt(2, 1))
    assert not(flt(1, 1))

def test_fle():
    assert fle(1, 2)
    assert fle(1, 1)
    assert not(fle(2, 1))

def test_fge():
    assert fge(2, 1)
    assert fge(2, 2)
    assert not(fge(1, 2))

def test_ie():
    with ignore_exception(ZeroDivisionError):
        print(1 / 0)
