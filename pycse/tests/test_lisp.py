from pycse.lisp import *

def test_symbol():
    assert Symbol('setf').lisp == 'setf'

def test_quote():
    assert Quote('setf').lisp == "'setf"

def test_sharpquote():
    assert SharpQuote('setf').lisp == "#'setf"

def test_cons():
    assert Cons('a', 3).lisp == '("a" . 3)'

def test_Alist():
    assert Alist(["a", 1, "b", 2]).lisp == '(("a" . 1) ("b" . 2))'

def test_vector():
    assert Vector(["a", 1, 3]).lisp == '["a" 1 3]'

def test_Comma():
    assert Comma(Symbol("setf")).lisp == ',setf'

def test_splice():
    assert Splice([1, 3]).lisp == ',@(1 3)'

def test_backquote():
    assert Backquote([Symbol("a"), 1]).lisp == '`(a 1)'

def test_comment():
    assert Comment(Symbol("test")).lisp == '; test'
