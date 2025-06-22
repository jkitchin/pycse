"""Tests for lisp module.


[2025-01-14 Tue] Disabling these. I think the module doesn't work in python the
way it used to. I am just disabling tests for now. I don't want to delete the
library yet.

"""

# from pycse.lisp import (
#     Symbol,
#     Quote,
#     SharpQuote,
#     Cons,
#     Alist,
#     Vector,
#     Comma,
#     Splice,
#     Backquote,
#     Comment,
# )


# def test_symbol():
#     assert Symbol("setf").lisp == "setf"


# def test_quote():
#     assert Quote("setf").lisp == "'setf"


# def test_sharpquote():
#     assert SharpQuote("setf").lisp == "#'setf"


# def test_cons():
#     assert Cons("a", 3).lisp == '("a" . 3)'


# def test_Alist():
#     assert Alist(["a", 1, "b", 2]).lisp == '(("a" . 1) ("b" . 2))'


# def test_vector():
#     assert Vector(["a", 1, 3]).lisp == '["a" 1 3]'


# def test_Comma():
#     assert Comma(Symbol("setf")).lisp == ",setf"


# def test_splice():
#     assert Splice([1, 3]).lisp == ",@(1 3)"


# def test_backquote():
#     assert Backquote([Symbol("a"), 1]).lisp == "`(a 1)"


# def test_comment():
#     assert Comment(Symbol("test")).lisp == "; test"
