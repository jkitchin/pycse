"""Library to convert Python data structures to lisp data structures.

This module adds a lisp property to the basic Python types which returns a
string of the data type in Lisp.


The module also provides some classes to help with more sophisticated data structures like alists, cons cells

Here are the transformations:

string     -> "string"
int, float -> number
[1, 2, 3]  -> (1 2 3)
(1, 2, 3)  -> (1 2 3)
{"a": 6}   -> (:a 6)  p-list

Symbol("lambda")     -> lambda
Cons(a, b)           -> (a . b)
Alist(["A" 2 "B" 5]) -> (("A" . 2) ("B" 5))
Quote("symbol")      -> 'symbol
Quote([1, 2, 3])     -> '(1 2 3)
SharpQuote("symbol") -> #'symbol
Vector([1, 2, 3])    -> [1 2 3]

You should be able to nest these to make complex programs. If you use custom data structures/classes, they need to have a lisp property defined.

http://kitchingroup.cheme.cmu.edu/blog/2015/05/16/Python-data-structures-to-lisp/

"""

import ctypes as c
import numpy as np

class PyObject_HEAD(c.Structure):
    _fields_ = [('HEAD', c.c_ubyte * (object.__basicsize__ -
                                      c.sizeof(c.c_void_p))),
                ('ob_type', c.c_void_p)]

_get_dict = c.pythonapi._PyObject_GetDictPtr
_get_dict.restype = c.POINTER(c.py_object)
_get_dict.argtypes = [c.py_object]


def get_dict(object):
    return _get_dict(object).contents.value

# This is how we convert simple types to lisp. Strings go in quotes, and numbers
# basically self-evaluate. These never contain other types.
get_dict(str)['lisp'] = property(lambda s:'"{}"'.format(str(s)))
get_dict(float)['lisp'] = property(lambda f:'{}'.format(str(f)))
get_dict(int)['lisp'] = property(lambda f:'{}'.format(str(f)))
get_dict(np.int64)['lisp'] = property(lambda f:'{}'.format(str(f)))


def lispify(L):
    """Convert a Python object L to a lisp representation."""
    if (isinstance(L, str)
        or isinstance(L, float)
        or isinstance(L, int)
        or isinstance(L, np.int64)
    ):
        return L.lisp
    elif (isinstance(L, list)
          or isinstance(L, tuple)
          or isinstance(L, np.ndarray)

    ):
        s = [element.lisp for element in L]
        return '(' + ' '.join(s) + ')'
    elif isinstance(L, dict):
        s = [":{0} {1}".format(key, val.lisp) for key, val in L.items()]
        return '(' + ' '.join(s) + ')'


get_dict(list)['lisp'] = property(lispify)
get_dict(tuple)['lisp'] = property(lispify)
get_dict(dict)['lisp'] = property(lispify)
get_dict(np.ndarray)['lisp'] = property(lispify)


# Some tools for generating lisp code

class Symbol(object):
    """A lisp symbol.

    This is used to print strings that do not have double quotes.
    """
    def __init__(self, sym):
        assert isinstance(sym, str)
        self.sym = sym
    @property
    def lisp(self):
        return self.sym

    def __str__(self):
        return self.lisp


class Quote(object):
    """Used to quote a symbol or form."""
    def __init__(self, sym):
        self.sym = sym

    @property
    def lisp(self):
        if isinstance(self.sym, str):
            s = self.sym
        else:
            # This is a list/vector
            s = self.sym.lisp
        return "'{}".format(s)

    def __str__(self):
        return self.lisp

class SharpQuote(object):
    """Used to SharpQuote a symbol or form."""
    def __init__(self, sym):
        self.sym = sym

    @property
    def lisp(self):
        if isinstance(self.sym, str):
            s = self.sym
        else:
            s = self.sym.lisp
        return "#'{}".format(s)

    def __str__(self):
        return self.lisp


class Cons(object):
    """A cons cell."""
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    @property
    def lisp(self):
        return '({} . {})'.format(self.car.lisp,
                                  self.cdr.lisp)

    def __str__(self):
        return self.lisp


class Alist(object):
    """A lisp association list."""
    def __init__(self, lst):
        self.list = lst

    @property
    def lisp(self):
        keys = self.list[0::2]
        vals = self.list[1::2]
        alist = [Cons(key, val) for key, val in zip(keys, vals)]
        return alist.lisp

    def __str__(self):
        return self.lisp

class Vector(object):
    """A lisp vector."""
    def __init__(self, lst):
        self.list = lst

    @property
    def lisp(self):
        return "[{}]".format(' '.join([x.lisp for x in self.list]))

    def __str__(self):
        return self.lisp


class Comma(object):
    """The comma operator."""
    def __init__(self, form):
        self.form = form

    @property
    def lisp(self):
        return ',{}'.format(self.form.lisp)

    def __str__(self):
        return self.lisp


class Splice(object):
    """A Splice object."""
    def __init__(self, form):
        self.form = form

    @property
    def lisp(self):
        return ',@{}'.format(self.form.lisp)

    def __str__(self):
        return self.lisp


class Backquote(object):
    """A Backquoted item."""
    def __init__(self, form):
        self.form = form

    @property
    def lisp(self):
        return '`{}'.format(self.form.lisp)

    def __str__(self):
        return self.lisp


class Comment(object):
    """A commented item in lisp."""
    def __init__(self, s):
        self.s = s

    @property
    def lisp(self):
        return '; {}'.format(self.s.lisp)

    def __str__(self):
        return self.lisp
