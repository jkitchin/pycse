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

    # Using the L wrapper
    str(L([1, 2, 3]))        # "(1 2 3)"
    str(L({"a": 6}))         # "(:a 6)"
    print(L([1, 2, 3]))      # "(1 2 3)"

    # Using helper classes
    str(Symbol("lambda"))    # "lambda"
    str(Quote("symbol"))     # "'symbol"

    # Concatenating with + operator
    Symbol("defun") + Symbol("square")  # "defun square"
    L([1, 2]) + Symbol("x")              # "(1 2) x"

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

    # Handle custom Lisp classes (L, Symbol, Quote, etc.) - use their __str__ method
    elif hasattr(obj, "__module__") and obj.__module__ == "pycse.lisp":
        return str(obj)

    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to Lisp: {obj}")


class L:
    """Lightweight wrapper for converting Python objects to Lisp representation.

    Usage:
        str(L([1, 2, 3]))        # "(1 2 3)"
        str(L({"a": 6}))         # "(:a 6)"
        print(L([1, 2, 3]))      # "(1 2 3)"
        L([1, 2]) + Symbol("x")  # "(1 2) x"
    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        """Initialize wrapper with a Python object."""
        self._obj = obj

    def __str__(self):
        """Return Lisp representation as string."""
        return to_lisp(self._obj)

    def __repr__(self):
        """Return readable representation."""
        return f"L({self._obj!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


# Helper classes for generating Lisp code


class Symbol:
    """A Lisp symbol.

    Symbols are used to print strings without double quotes.

    Usage:
        str(Symbol("lambda"))    # "lambda"
        Symbol("defun") + Symbol("x")  # "defun x"
    """

    def __init__(self, sym):
        """Initialize a Symbol."""
        if not isinstance(sym, str):
            raise TypeError(f"Symbol must be a string, not {type(sym).__name__}")
        self.sym = sym

    def __str__(self):
        """Return Lisp representation of symbol."""
        return self.sym

    def __repr__(self):
        """Return readable representation."""
        return f"Symbol({self.sym!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Quote:
    """Quote a symbol or form.

    Usage:
        str(Quote("symbol"))        # "'symbol"
        str(Quote([1, 2, 3]))       # "'(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a Quote."""
        self.form = form

    def __str__(self):
        """Return Lisp representation with quote prefix."""
        if isinstance(self.form, str):
            return f"'{self.form}"
        else:
            return f"'{to_lisp(self.form)}"

    def __repr__(self):
        """Return readable representation."""
        return f"Quote({self.form!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class SharpQuote:
    """Function quote (#') a symbol or form.

    Usage:
        str(SharpQuote("lambda"))   # "#'lambda"
    """

    def __init__(self, form):
        """Initialize a SharpQuote."""
        self.form = form

    def __str__(self):
        """Return Lisp representation with #' prefix."""
        if isinstance(self.form, str):
            return f"#'{self.form}"
        else:
            return f"#'{to_lisp(self.form)}"

    def __repr__(self):
        """Return readable representation."""
        return f"SharpQuote({self.form!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Cons:
    """A cons cell (dotted pair).

    Usage:
        str(Cons("a", "b"))         # "(a . b)"
        str(Cons(1, 2))             # "(1 . 2)"
    """

    def __init__(self, car, cdr):
        """Initialize a Cons cell with car and cdr."""
        self.car = car
        self.cdr = cdr

    def __str__(self):
        """Return Lisp representation as dotted pair."""
        return f"({to_lisp(self.car)} . {to_lisp(self.cdr)})"

    def __repr__(self):
        """Return readable representation."""
        return f"Cons({self.car!r}, {self.cdr!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Alist:
    """A Lisp association list.

    Usage:
        str(Alist(["A", 2, "B", 5]))    # '(("A" . 2) ("B" . 5))'
    """

    def __init__(self, lst):
        """Initialize an alist from a flat list [key1, val1, key2, val2, ...]."""
        if len(lst) % 2 != 0:
            raise ValueError("Alist requires an even number of elements")
        self.list = lst

    def __str__(self):
        """Return Lisp representation as association list."""
        keys = self.list[0::2]
        vals = self.list[1::2]
        pairs = [Cons(key, val) for key, val in zip(keys, vals)]
        return to_lisp(pairs)

    def __repr__(self):
        """Return readable representation."""
        return f"Alist({self.list!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Vector:
    """A Lisp vector.

    Usage:
        str(Vector([1, 2, 3]))      # "[1 2 3]"
    """

    def __init__(self, lst):
        """Initialize a vector from a list."""
        self.list = lst

    def __str__(self):
        """Return Lisp representation as vector."""
        if len(self.list) == 0:
            return "[]"
        elements = [to_lisp(x) for x in self.list]
        return "[" + " ".join(elements) + "]"

    def __repr__(self):
        """Return readable representation."""
        return f"Vector({self.list!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Backquote:
    """A backquoted form.

    Usage:
        str(Backquote([1, 2, 3]))   # "`(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a backquote."""
        self.form = form

    def __str__(self):
        """Return Lisp representation with backquote prefix."""
        return f"`{to_lisp(self.form)}"

    def __repr__(self):
        """Return readable representation."""
        return f"Backquote({self.form!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Comma:
    """The comma (unquote) operator.

    Usage:
        str(Comma(Symbol("x")))     # ",x"
    """

    def __init__(self, form):
        """Initialize a comma operator."""
        self.form = form

    def __str__(self):
        """Return Lisp representation with comma prefix."""
        return f",{to_lisp(self.form)}"

    def __repr__(self):
        """Return readable representation."""
        return f"Comma({self.form!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Splice:
    """The splice (,@) operator.

    Usage:
        str(Splice([1, 2, 3]))      # ",@(1 2 3)"
    """

    def __init__(self, form):
        """Initialize a splice operator."""
        self.form = form

    def __str__(self):
        """Return Lisp representation with ,@ prefix."""
        return f",@{to_lisp(self.form)}"

    def __repr__(self):
        """Return readable representation."""
        return f"Splice({self.form!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"


class Comment:
    """A comment in Lisp.

    Usage:
        str(Comment("This is a comment"))   # "; This is a comment"
    """

    def __init__(self, text):
        """Initialize a comment with text."""
        self.text = text

    def __str__(self):
        """Return Lisp representation as comment."""
        return f"; {to_lisp(self.text) if not isinstance(self.text, str) else self.text}"

    def __repr__(self):
        """Return readable representation."""
        return f"Comment({self.text!r})"

    def __add__(self, other):
        """Concatenate Lisp representations with space separator."""
        return f"{str(self)} {str(other)}"
