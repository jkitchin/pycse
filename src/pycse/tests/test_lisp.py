"""Tests for lisp module."""

import pytest
import numpy as np
from pycse.lisp import (
    to_lisp,
    L,
    Symbol,
    Quote,
    SharpQuote,
    Cons,
    Alist,
    Vector,
    Backquote,
    Comma,
    Splice,
    Comment,
)


# Tests for to_lisp() function
class TestToLisp:
    """Tests for the to_lisp conversion function."""

    def test_string_conversion(self):
        """Test string conversion to Lisp."""
        assert to_lisp("hello") == '"hello"'
        assert to_lisp("") == '""'
        assert to_lisp("with spaces") == '"with spaces"'

    def test_int_conversion(self):
        """Test integer conversion to Lisp."""
        assert to_lisp(42) == "42"
        assert to_lisp(0) == "0"
        assert to_lisp(-10) == "-10"

    def test_float_conversion(self):
        """Test float conversion to Lisp."""
        assert to_lisp(3.14) == "3.14"
        assert to_lisp(0.0) == "0.0"
        assert to_lisp(-2.5) == "-2.5"

    def test_bool_conversion(self):
        """Test boolean conversion to Lisp."""
        assert to_lisp(True) == "t"
        assert to_lisp(False) == "nil"

    def test_none_conversion(self):
        """Test None conversion to Lisp."""
        assert to_lisp(None) == "nil"

    def test_list_conversion(self):
        """Test list conversion to Lisp."""
        assert to_lisp([1, 2, 3]) == "(1 2 3)"
        assert to_lisp([]) == "()"
        assert to_lisp(["a", "b", "c"]) == '("a" "b" "c")'

    def test_tuple_conversion(self):
        """Test tuple conversion to Lisp."""
        assert to_lisp((1, 2, 3)) == "(1 2 3)"
        assert to_lisp(()) == "()"

    def test_dict_conversion(self):
        """Test dict conversion to Lisp (as property list)."""
        result = to_lisp({"a": 6})
        assert result == "(:a 6)"

        result = to_lisp({})
        assert result == "()"

    def test_dict_multi_key(self):
        """Test dict with multiple keys."""
        result = to_lisp({"x": 1, "y": 2})
        # Dict order may vary, check both possibilities
        assert result in ["(:x 1 :y 2)", "(:y 2 :x 1)"]

    def test_numpy_array_conversion(self):
        """Test numpy array conversion to Lisp."""
        arr = np.array([1, 2, 3])
        assert to_lisp(arr) == "(1 2 3)"

    def test_numpy_int_conversion(self):
        """Test numpy integer conversion to Lisp."""
        assert to_lisp(np.int64(42)) == "42"
        assert to_lisp(np.int32(10)) == "10"

    def test_numpy_float_conversion(self):
        """Test numpy float conversion to Lisp."""
        assert to_lisp(np.float64(3.14)) == "3.14"

    def test_nested_list_conversion(self):
        """Test nested list conversion to Lisp."""
        assert to_lisp([1, [2, 3], 4]) == "(1 (2 3) 4)"
        assert to_lisp([[1, 2], [3, 4]]) == "((1 2) (3 4))"

    def test_mixed_types_conversion(self):
        """Test conversion of mixed types."""
        assert to_lisp([1, "hello", 3.14]) == '(1 "hello" 3.14)'

    def test_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            to_lisp(object())

    def test_custom_lisp_objects(self):
        """Test that custom Lisp objects convert correctly with to_lisp()."""
        sym = Symbol("lambda")
        assert to_lisp(sym) == "lambda"


# Tests for L wrapper class
class TestLWrapper:
    """Tests for the L wrapper class."""

    def test_l_wrapper_basic(self):
        """Test L wrapper basic functionality."""
        assert str(L([1, 2, 3])) == "(1 2 3)"
        assert str(L("hello")) == '"hello"'
        assert str(L(42)) == "42"

    def test_l_wrapper_str(self):
        """Test L wrapper __str__ method."""
        assert str(L([1, 2, 3])) == "(1 2 3)"

    def test_l_wrapper_repr(self):
        """Test L wrapper __repr__ method."""
        assert repr(L([1, 2, 3])) == "L([1, 2, 3])"

    def test_l_wrapper_dict(self):
        """Test L wrapper with dictionary."""
        assert str(L({"a": 6})) == "(:a 6)"

    def test_l_wrapper_nested(self):
        """Test L wrapper with nested structures."""
        assert str(L([[1, 2], [3, 4]])) == "((1 2) (3 4))"

    def test_l_wrapper_add(self):
        """Test L wrapper __add__ method for concatenation."""
        assert L([1, 2]) + Symbol("x") == "(1 2) x"
        assert L([1, 2]) + L([3, 4]) == "(1 2) (3 4)"
        assert L("test") + Symbol("foo") == '"test" foo'


# Tests for Symbol class
class TestSymbol:
    """Tests for Symbol class."""

    def test_symbol_basic(self):
        """Test Symbol basic functionality."""
        assert str(Symbol("lambda")) == "lambda"
        assert str(Symbol("defun")) == "defun"
        assert str(Symbol("setf")) == "setf"

    def test_symbol_str(self):
        """Test Symbol __str__ method."""
        assert str(Symbol("lambda")) == "lambda"

    def test_symbol_repr(self):
        """Test Symbol __repr__ method."""
        assert repr(Symbol("lambda")) == "Symbol('lambda')"

    def test_symbol_non_string_error(self):
        """Test Symbol raises error for non-string."""
        with pytest.raises(TypeError, match="Symbol must be a string"):
            Symbol(123)

    def test_symbol_in_list(self):
        """Test Symbol used in list conversion."""
        result = to_lisp([Symbol("lambda"), Symbol("x"), 1])
        assert result == "(lambda x 1)"


# Tests for Quote class
class TestQuote:
    """Tests for Quote class."""

    def test_quote_string(self):
        """Test Quote with string."""
        assert str(Quote("symbol")) == "'symbol"
        assert str(Quote("setf")) == "'setf"

    def test_quote_list(self):
        """Test Quote with list."""
        assert str(Quote([1, 2, 3])) == "'(1 2 3)"

    def test_quote_str(self):
        """Test Quote __str__ method."""
        assert str(Quote("x")) == "'x"

    def test_quote_repr(self):
        """Test Quote __repr__ method."""
        assert repr(Quote("x")) == "Quote('x')"


# Tests for SharpQuote class
class TestSharpQuote:
    """Tests for SharpQuote class."""

    def test_sharpquote_string(self):
        """Test SharpQuote with string."""
        assert str(SharpQuote("lambda")) == "#'lambda"
        assert str(SharpQuote("setf")) == "#'setf"

    def test_sharpquote_list(self):
        """Test SharpQuote with list."""
        assert str(SharpQuote([1, 2])) == "#'(1 2)"

    def test_sharpquote_str(self):
        """Test SharpQuote __str__ method."""
        assert str(SharpQuote("func")) == "#'func"

    def test_sharpquote_repr(self):
        """Test SharpQuote __repr__ method."""
        assert repr(SharpQuote("func")) == "SharpQuote('func')"


# Tests for Cons class
class TestCons:
    """Tests for Cons class."""

    def test_cons_basic(self):
        """Test Cons basic functionality."""
        assert str(Cons("a", "b")) == '("a" . "b")'
        assert str(Cons(1, 2)) == "(1 . 2)"
        assert str(Cons("a", 3)) == '("a" . 3)'

    def test_cons_str(self):
        """Test Cons __str__ method."""
        assert str(Cons("x", "y")) == '("x" . "y")'

    def test_cons_repr(self):
        """Test Cons __repr__ method."""
        assert repr(Cons("a", "b")) == "Cons('a', 'b')"

    def test_cons_nested(self):
        """Test Cons with nested structures."""
        assert str(Cons([1, 2], [3, 4])) == "((1 2) . (3 4))"


# Tests for Alist class
class TestAlist:
    """Tests for Alist class."""

    def test_alist_basic(self):
        """Test Alist basic functionality."""
        result = str(Alist(["A", 2, "B", 5]))
        assert result == '(("A" . 2) ("B" . 5))'

        result = str(Alist(["a", 1, "b", 2]))
        assert result == '(("a" . 1) ("b" . 2))'

    def test_alist_str(self):
        """Test Alist __str__ method."""
        result = str(Alist(["x", 1, "y", 2]))
        assert result == '(("x" . 1) ("y" . 2))'

    def test_alist_repr(self):
        """Test Alist __repr__ method."""
        assert repr(Alist(["a", 1])) == "Alist(['a', 1])"

    def test_alist_odd_elements_error(self):
        """Test Alist raises error for odd number of elements."""
        with pytest.raises(ValueError, match="even number"):
            Alist([1, 2, 3])

    def test_alist_empty(self):
        """Test Alist with empty list."""
        assert str(Alist([])) == "()"


# Tests for Vector class
class TestVector:
    """Tests for Vector class."""

    def test_vector_basic(self):
        """Test Vector basic functionality."""
        assert str(Vector([1, 2, 3])) == "[1 2 3]"
        assert str(Vector(["a", 1, 3])) == '["a" 1 3]'

    def test_vector_empty(self):
        """Test Vector with empty list."""
        assert str(Vector([])) == "[]"

    def test_vector_str(self):
        """Test Vector __str__ method."""
        assert str(Vector([1, 2])) == "[1 2]"

    def test_vector_repr(self):
        """Test Vector __repr__ method."""
        assert repr(Vector([1, 2])) == "Vector([1, 2])"

    def test_vector_mixed_types(self):
        """Test Vector with mixed types."""
        assert str(Vector([1, "a", 3.14])) == '[1 "a" 3.14]'


# Tests for Backquote class
class TestBackquote:
    """Tests for Backquote class."""

    def test_backquote_basic(self):
        """Test Backquote basic functionality."""
        assert str(Backquote([1, 2, 3])) == "`(1 2 3)"
        assert str(Backquote([Symbol("a"), 1])) == "`(a 1)"

    def test_backquote_str(self):
        """Test Backquote __str__ method."""
        assert str(Backquote([1, 2])) == "`(1 2)"

    def test_backquote_repr(self):
        """Test Backquote __repr__ method."""
        assert repr(Backquote([1, 2])) == "Backquote([1, 2])"


# Tests for Comma class
class TestComma:
    """Tests for Comma class."""

    def test_comma_basic(self):
        """Test Comma basic functionality."""
        assert str(Comma(Symbol("x"))) == ",x"
        assert str(Comma(Symbol("setf"))) == ",setf"

    def test_comma_str(self):
        """Test Comma __str__ method."""
        assert str(Comma(Symbol("y"))) == ",y"

    def test_comma_repr(self):
        """Test Comma __repr__ method."""
        result = repr(Comma(Symbol("x")))
        assert "Comma" in result


# Tests for Splice class
class TestSplice:
    """Tests for Splice class."""

    def test_splice_basic(self):
        """Test Splice basic functionality."""
        assert str(Splice([1, 2, 3])) == ",@(1 2 3)"
        assert str(Splice([1, 3])) == ",@(1 3)"

    def test_splice_str(self):
        """Test Splice __str__ method."""
        assert str(Splice([1, 2])) == ",@(1 2)"

    def test_splice_repr(self):
        """Test Splice __repr__ method."""
        assert "Splice" in repr(Splice([1, 2]))


# Tests for Comment class
class TestComment:
    """Tests for Comment class."""

    def test_comment_basic(self):
        """Test Comment basic functionality."""
        assert str(Comment("This is a comment")) == "; This is a comment"
        assert str(Comment("test")) == "; test"

    def test_comment_with_symbol(self):
        """Test Comment with Symbol."""
        assert str(Comment(Symbol("test"))) == "; test"

    def test_comment_str(self):
        """Test Comment __str__ method."""
        assert str(Comment("test")) == "; test"

    def test_comment_repr(self):
        """Test Comment __repr__ method."""
        assert repr(Comment("test")) == "Comment('test')"


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_nested_structure(self):
        """Test complex nested Lisp structure."""
        result = to_lisp(
            [
                Symbol("defun"),
                Symbol("add"),
                [Symbol("x"), Symbol("y")],
                [Symbol("+"), Symbol("x"), Symbol("y")],
            ]
        )
        assert result == "(defun add (x y) (+ x y))"

    def test_quoted_list(self):
        """Test quoted list."""
        assert str(Quote([1, 2, 3])) == "'(1 2 3)"

    def test_backquoted_with_comma(self):
        """Test backquote with comma (unquote)."""
        result = to_lisp([Backquote([Symbol("list"), Comma(Symbol("x")), 2])])
        assert result == "(`(list ,x 2))"

    def test_function_definition(self):
        """Test Lisp function definition structure."""
        defun = [
            Symbol("defun"),
            Symbol("square"),
            [Symbol("x")],
            [Symbol("*"), Symbol("x"), Symbol("x")],
        ]
        assert to_lisp(defun) == "(defun square (x) (* x x))"

    def test_l_wrapper_with_symbols(self):
        """Test L wrapper with Symbol objects."""
        result = str(L([Symbol("lambda"), [Symbol("x")], Symbol("x")]))
        assert result == "(lambda (x) x)"

    def test_mixed_wrapper_and_helpers(self):
        """Test mixing L wrapper with helper classes."""
        result = str(L([Quote("symbol"), Vector([1, 2, 3])]))
        assert result == "('symbol [1 2 3])"

    def test_add_operations(self):
        """Test __add__() method across different Lisp classes."""
        # Test Symbol + Symbol
        assert Symbol("defun") + Symbol("square") == "defun square"

        # Test Quote + other
        assert Quote("x") + Symbol("y") == "'x y"

        # Test Cons + other
        assert Cons("a", "b") + Cons("c", "d") == '("a" . "b") ("c" . "d")'

        # Test Vector + other
        assert Vector([1, 2]) + Vector([3, 4]) == "[1 2] [3 4]"

        # Test L + Symbol
        assert L([Symbol("x")]) + Symbol("y") == "(x) y"

        # Test building complex expressions with explicit concatenation
        assert Symbol("defun") + L([Symbol("x"), Symbol("y")]) == "defun (x y)"
