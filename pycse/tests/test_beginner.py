from pycse.beginner import *
from nose.tools import *

a = [1, 2, 3, 4, 5]

def test_first_a():
    assert first(a) == 1

@raises(Exception)
def test_first_b():
    assert first(1) == 1


def test_second_a():
    assert second(a) == 2

@raises(Exception)
def test_second_b ():
    assert second(1) == 0

@raises(Exception)
def test_second_c ():
    assert second([1]) == 0

def test_third_a():
    assert third(a) == 3

@raises(Exception)
def test_third_b ():
    assert third(1) == 0

@raises(Exception)
def test_third_c ():
    assert third([1, 2]) == 0


def test_fourth_a():
    assert fourth(a) == 4

@raises(Exception)
def test_fourth_b ():
    assert fourth(1) == 0

@raises(Exception)
def test_fourth_c ():
    assert fourth([1]) == 0

def test_fifth_a():
    assert fifth(a) == 5

@raises(Exception)
def test_fifth_b ():
    assert fifth(1) == 0

@raises(Exception)
def test_fifth_c ():
    assert fifth([1]) == 0


def test_rest():
    assert rest(a) == [2, 3, 4, 5]

def test_last():
    assert last(a) == 5

@raises(Exception)
def test_last_b():
    assert last(5)

def test_butlast():
    assert butlast(a) == [1, 2, 3, 4]

def test_nth():
    assert nth(a, 1) == 2

@raises(Exception)
def test_nth_a():
    assert nth(a, 6) == None


def test_cut_a():
    print(cut(a, 1, 2))
    assert cut(a, 1, None, 2) == [2, 4]

def test_integrate():
    assert integrate(lambda x: 1, 0, 1) == 1

from pycse import feq

def test_nsolve():
    assert feq(nsolve(lambda x: x - 3, 2), 3)

@raises(Exception)
def test_nsolve_2():
    assert feq(nsolve(lambda x: x + 3, 2), 3)


def test_nsolve_3():
    assert (nsolve(lambda X: [X[0] - 3, X[1] - 4], [2, 2]) == [3, 4]).all()

def test_heaviside_1():
    assert feq(0, heaviside(-1))

def test_heaviside_2():
    assert feq(1, heaviside(1))

def test_heaviside_3():
    assert feq(0.5, heaviside(0))
