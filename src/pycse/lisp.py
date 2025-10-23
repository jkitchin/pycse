"""Library to convert Python data structures to lisp data structures.

This module provides a lightweight wrapper and helper classes to convert
Python objects to Lisp representations.

Transformations:
    string     -> "string"
    int, float -> number
    [1, 2, 3]  -> (1 2 3)
    (1, 2, 3)  -> (1 2 3)
    {"a": 6}   -> (:a 6)  p-list

Helper classes:
    Symbol("lambda")     -> lambda
    Cons(a, b)           -> (a . b)
    Alist(["A" 2 "B" 5]) -> (("A" . 2) ("B" 5))
    Quote("symbol")      -> 'symbol
    Quote([1, 2, 3])     -> '(1 2 3)
    SharpQuote("symbol") -> #'symbol
    Vector([1, 2, 3])    -> [1 2 3]

Usage:
    from pycse.lisp import L, Symbol, Quote

    # Using the wrapper for .lisp attribute
    L([1, 2, 3]).lisp        # "(1 2 3)"
    L({"a": 6}).lisp         # "(:a 6)"
    print(L([1, 2, 3]))      # "(1 2 3)"

    # Using helper classes
    Symbol("lambda").lisp    # "lambda"
    Quote("symbol").lisp     # "'symbol"

http://kitchingroup.cheme.cmu.edu/blog/2015/05/16/Python-data-structures-to-lisp/
"""

import numpy as np


def to_lisp(obj):
    """Convert a Python object to a Lisp representation string.

    Args:
        obj: Python object to convert (str, int, float, list, tuple, dict, etc.)

    Returns:
        str: Lisp representation of the object

    Raises:
        TypeError: If object cannot be converted to Lisp
    """
    # Handle objects with .lisp property (custom classes)
    if hasattr(obj, "lisp"):
        return obj.lisp

    # Handle basic types
    if isinstance(obj, str):
        return f'"{obj}"'
    elif isinstance(obj, bool):
        # Handle bool before int (bool is subclass of int)
        return "t" if obj else "nil"
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        return str(obj)

    # Handle collections
    elif isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) == 0:
            return "()"
        elements = [to_lisp(item) for item in obj]
        return "(" + " ".join(elements) + ")"

    # Handle dictionaries as property lists
    elif isinstance(obj, dict):
        if len(obj) == 0:
            return "()"
        items = [f":{key} {to_lisp(value)}" for key, value in obj.items()]
        return "(" + " ".join(items) + ")"

    # Handle None
    elif obj is None:
        return "nil"

    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to Lisp: {obj}")


class L:
    """Lightweight wrapper that provides .lisp attribute for any Python object.

    Usage:
        L([1, 2, 3]).lisp        # "(1 2 3)"
        L({"a": 6}).lisp         # "(:a 6)"
        print(L([1, 2, 3]))      # "(1 2 3)"
    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        """Initialize wrapper with a Python object."""
        self._obj = obj

    @property
    def lisp(self):
        """Return Lisp representation of wrapped object."""
        return to_lisp(self._obj)

    def __str__(self):
        """Return Lisp representation as string."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"L({self._obj!r})"


# Helper classes for generating Lisp code


class Symbol:
    """A Lisp symbol.

    Symbols are used to print strings without double quotes.

    Usage:
        Symbol("lambda").lisp    # "lambda"
        Symbol("defun").lisp     # "defun"
    """

    def __init__(self, sym):
        """Initialize a Symbol."""
        if not isinstance(sym, str):
            raise TypeError(f"Symbol must be a string, not {type(sym).__name__}")
        self.sym = sym

    @property
    def lisp(self):
        """Return Lisp representation of symbol."""
        return self.sym

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Symbol({self.sym!r})"


class Quote:
    """Quote a symbol or form.

    Usage:
        Quote("symbol").lisp        # "'symbol"
        Quote([1, 2, 3]).lisp       # "'(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a Quote."""
        self.form = form

    @property
    def lisp(self):
        """Return Lisp representation with quote prefix."""
        if isinstance(self.form, str):
            return f"'{self.form}"
        else:
            return f"'{to_lisp(self.form)}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Quote({self.form!r})"


class SharpQuote:
    """Function quote (#') a symbol or form.

    Usage:
        SharpQuote("lambda").lisp   # "#'lambda"
    """

    def __init__(self, form):
        """Initialize a SharpQuote."""
        self.form = form

    @property
    def lisp(self):
        """Return Lisp representation with #' prefix."""
        if isinstance(self.form, str):
            return f"#'{self.form}"
        else:
            return f"#'{to_lisp(self.form)}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"SharpQuote({self.form!r})"


class Cons:
    """A cons cell (dotted pair).

    Usage:
        Cons("a", "b").lisp         # "(a . b)"
        Cons(1, 2).lisp             # "(1 . 2)"
    """

    def __init__(self, car, cdr):
        """Initialize a Cons cell with car and cdr."""
        self.car = car
        self.cdr = cdr

    @property
    def lisp(self):
        """Return Lisp representation as dotted pair."""
        return f"({to_lisp(self.car)} . {to_lisp(self.cdr)})"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Cons({self.car!r}, {self.cdr!r})"


class Alist:
    """A Lisp association list.

    Usage:
        Alist(["A", 2, "B", 5]).lisp    # '(("A" . 2) ("B" . 5))'
    """

    def __init__(self, lst):
        """Initialize an alist from a flat list [key1, val1, key2, val2, ...]."""
        if len(lst) % 2 != 0:
            raise ValueError("Alist requires an even number of elements")
        self.list = lst

    @property
    def lisp(self):
        """Return Lisp representation as association list."""
        keys = self.list[0::2]
        vals = self.list[1::2]
        pairs = [Cons(key, val) for key, val in zip(keys, vals)]
        return to_lisp(pairs)

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Alist({self.list!r})"


class Vector:
    """A Lisp vector.

    Usage:
        Vector([1, 2, 3]).lisp      # "[1 2 3]"
    """

    def __init__(self, lst):
        """Initialize a vector from a list."""
        self.list = lst

    @property
    def lisp(self):
        """Return Lisp representation as vector."""
        if len(self.list) == 0:
            return "[]"
        elements = [to_lisp(x) for x in self.list]
        return "[" + " ".join(elements) + "]"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Vector({self.list!r})"


class Backquote:
    """A backquoted form.

    Usage:
        Backquote([1, 2, 3]).lisp   # "`(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a backquote."""
        self.form = form

    @property
    def lisp(self):
        """Return Lisp representation with backquote prefix."""
        return f"`{to_lisp(self.form)}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Backquote({self.form!r})"


class Comma:
    """The comma (unquote) operator.

    Usage:
        Comma(Symbol("x")).lisp     # ",x"
    """

    def __init__(self, form):
        """Initialize a comma operator."""
        self.form = form

    @property
    def lisp(self):
        """Return Lisp representation with comma prefix."""
        return f",{to_lisp(self.form)}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Comma({self.form!r})"


class Splice:
    """The splice (,@) operator.

    Usage:
        Splice([1, 2, 3]).lisp      # ",@(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a splice operator."""
        self.form = form

    @property
    def lisp(self):
        """Return Lisp representation with ,@ prefix."""
        return f",@{to_lisp(self.form)}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Splice({self.form!r})"


class Comment:
    """A comment in Lisp.

    Usage:
        Comment("This is a comment").lisp   # "; This is a comment"
    """

    def __init__(self, text):
        """Initialize a comment with text."""
        self.text = text

    @property
    def lisp(self):
        """Return Lisp representation as comment."""
        return f"; {to_lisp(self.text) if not isinstance(self.text, str) else self.text}"

    def __str__(self):
        """Return string representation."""
        return self.lisp

    def __repr__(self):
        """Return readable representation."""
        return f"Comment({self.text!r})"
