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

    def test_object_with_lisp_property(self):
        """Test that objects with .lisp property use it."""
        sym = Symbol("lambda")
        assert to_lisp(sym) == "lambda"


# Tests for L wrapper class
class TestLWrapper:
    """Tests for the L wrapper class."""

    def test_l_wrapper_basic(self):
        """Test L wrapper basic functionality."""
        assert L([1, 2, 3]).lisp == "(1 2 3)"
        assert L("hello").lisp == '"hello"'
        assert L(42).lisp == "42"

    def test_l_wrapper_str(self):
        """Test L wrapper __str__ method."""
        assert str(L([1, 2, 3])) == "(1 2 3)"

    def test_l_wrapper_repr(self):
        """Test L wrapper __repr__ method."""
        assert repr(L([1, 2, 3])) == "L([1, 2, 3])"

    def test_l_wrapper_dict(self):
        """Test L wrapper with dictionary."""
        assert L({"a": 6}).lisp == "(:a 6)"

    def test_l_wrapper_nested(self):
        """Test L wrapper with nested structures."""
        assert L([[1, 2], [3, 4]]).lisp == "((1 2) (3 4))"


# Tests for Symbol class
class TestSymbol:
    """Tests for Symbol class."""

    def test_symbol_basic(self):
        """Test Symbol basic functionality."""
        assert Symbol("lambda").lisp == "lambda"
        assert Symbol("defun").lisp == "defun"
        assert Symbol("setf").lisp == "setf"

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
        assert Quote("symbol").lisp == "'symbol"
        assert Quote("setf").lisp == "'setf"

    def test_quote_list(self):
        """Test Quote with list."""
        assert Quote([1, 2, 3]).lisp == "'(1 2 3)"

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
        assert SharpQuote("lambda").lisp == "#'lambda"
        assert SharpQuote("setf").lisp == "#'setf"

    def test_sharpquote_list(self):
        """Test SharpQuote with list."""
        assert SharpQuote([1, 2]).lisp == "#'(1 2)"

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
        assert Cons("a", "b").lisp == '("a" . "b")'
        assert Cons(1, 2).lisp == "(1 . 2)"
        assert Cons("a", 3).lisp == '("a" . 3)'

    def test_cons_str(self):
        """Test Cons __str__ method."""
        assert str(Cons("x", "y")) == '("x" . "y")'

    def test_cons_repr(self):
        """Test Cons __repr__ method."""
        assert repr(Cons("a", "b")) == "Cons('a', 'b')"

    def test_cons_nested(self):
        """Test Cons with nested structures."""
        assert Cons([1, 2], [3, 4]).lisp == "((1 2) . (3 4))"


# Tests for Alist class
class TestAlist:
    """Tests for Alist class."""

    def test_alist_basic(self):
        """Test Alist basic functionality."""
        result = Alist(["A", 2, "B", 5]).lisp
        assert result == '(("A" . 2) ("B" . 5))'

        result = Alist(["a", 1, "b", 2]).lisp
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
        assert Alist([]).lisp == "()"


# Tests for Vector class
class TestVector:
    """Tests for Vector class."""

    def test_vector_basic(self):
        """Test Vector basic functionality."""
        assert Vector([1, 2, 3]).lisp == "[1 2 3]"
        assert Vector(["a", 1, 3]).lisp == '["a" 1 3]'

    def test_vector_empty(self):
        """Test Vector with empty list."""
        assert Vector([]).lisp == "[]"

    def test_vector_str(self):
        """Test Vector __str__ method."""
        assert str(Vector([1, 2])) == "[1 2]"

    def test_vector_repr(self):
        """Test Vector __repr__ method."""
        assert repr(Vector([1, 2])) == "Vector([1, 2])"

    def test_vector_mixed_types(self):
        """Test Vector with mixed types."""
        assert Vector([1, "a", 3.14]).lisp == '[1 "a" 3.14]'


# Tests for Backquote class
class TestBackquote:
    """Tests for Backquote class."""

    def test_backquote_basic(self):
        """Test Backquote basic functionality."""
        assert Backquote([1, 2, 3]).lisp == "`(1 2 3)"
        assert Backquote([Symbol("a"), 1]).lisp == "`(a 1)"

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
        assert Comma(Symbol("x")).lisp == ",x"
        assert Comma(Symbol("setf")).lisp == ",setf"

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
        assert Splice([1, 2, 3]).lisp == ",@(1 2 3)"
        assert Splice([1, 3]).lisp == ",@(1 3)"

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
        assert Comment("This is a comment").lisp == "; This is a comment"
        assert Comment("test").lisp == "; test"

    def test_comment_with_symbol(self):
        """Test Comment with Symbol."""
        assert Comment(Symbol("test")).lisp == "; test"

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
        assert Quote([1, 2, 3]).lisp == "'(1 2 3)"

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
        result = L([Symbol("lambda"), [Symbol("x")], Symbol("x")]).lisp
        assert result == "(lambda (x) x)"

    def test_mixed_wrapper_and_helpers(self):
        """Test mixing L wrapper with helper classes."""
        result = L([Quote("symbol"), Vector([1, 2, 3])]).lisp
        assert result == "('symbol [1 2 3])"
