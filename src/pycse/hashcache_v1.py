"""hashcache - a decorator for persistent, file/hash-based cache

I found some features of joblib were unsuitable for how I want to use a cache.

1. The "file" Python thinks the function is in is used to save the results in
joblib, which leads to repeated runs if you run the same code in Python,
notebook or stdin, and means the cache is not portable to other machines, and
maybe not even in time since temp directories and kernel parameters are
involved. I could not figure out how to change those in joblib.

2. joblib uses the function source code in the hash, so inconsequential changes
like whitespace, docstrings and comments change the hash.

This library aims to provide a simpler version of what I wish joblib did for me.

Results are cached based on a hash of the function name, argnames, bytecode, arg
values and kwarg values. I use joblib.hash for this. This means any two
functions with the same bytecode, even if they have different names, will cache
to the same result.

The cache location is set as a function attribute:

    hashcache.cache = './cache'


This is alpha, proof of concept code. Test it a lot for your use case. The API
is not stable, and subject to change.

Some things to do:

1. the function attributes are kind of weird, maybe these should be decorator
arguments.

Pros:

1. File-based cache which means many functions can run in parallel reading and
writing, and you are limited only by file io speeds, and disk space.

2. semi-portability. The cache could be synced across machines, and caches
can be merged with little risk of conflict.

3. No server is required. Everything is done at the OS level.

4. Extendability. You can define your own functions for loading and dumping
data.

Cons:

1. hashes are fragile and not robust. They are fragile with respect to any
changes in how byte-code is made, or via mutable arguments, etc. The hashes are
not robust to system level changes like library versions, or global variables.
The only advantage of hashes is you can compute them.

2. File-based cache which means if you generate thousands of files, it can be
slow to delete them. Although it should be fast to access the results since you
access them directly by path, it will not be fast to iterate over all the
results, e.g. if you want to implement some kind of search or reporting.

3. No server. You have to roll your own update strategy if you run things on
multiple machines that should all cache to a common location.

Changelog
---------

[2023-09-23 Sat] Changed hash signature (breaking change). It is too difficult
to figure out how to capture global state, and the use of internal variable
names is not consistent with using the bytecode to be insensitive to
unimportant variable name changes.

Pulled out some functions for loading and dumping data. This is a precursor to
enabling other backends like lmdb or sqlite instead of files. You can then
simply provide new functions for this.

"""

import functools
import inspect
import joblib
import os
from pathlib import Path
import pprint
import time


def get_standardized_args(func, args, kwargs):
    """Returns a standardized dictionary of kwargs for func(args, kwargs)

    This dictionary includes default values, even if they were not called.

    """
    sig = inspect.signature(func)
    standardized_args = sig.bind(*args, **kwargs)
    standardized_args.apply_defaults()
    return standardized_args.arguments


def get_hash(func, args, kwargs):
    """Get a hash for running FUNC(ARGS, KWARGS).

    This is the most critical feature of hashcache as it provides a key to store
    and look up results later. You should think carefully before changing this
    function, it breaks past caches.

    FUNC should be as pure as reasonable. This hash is insensitive to global
    variables.

    The hash is on the function name, bytecode, and a standardized kwargs
    including defaults. We use bytecode because it is insensitive to things like
    whitespace, comments, docstrings, and variable name changes that don't
    affect results. It is assumed that two functions with the same name and
    bytecode will evaluate to the same result.

    """
    return joblib.hash(
        [
            func.__code__.co_name,  # This is the function name
            func.__code__.co_code,  # this is the function bytecode
            get_standardized_args(func, args, kwargs),  # The args used, including defaults
        ],
        hash_name="sha1",
    )


def get_hashpath(hsh):
    """Return path to file for HSH."""
    cache = Path(hashcache.cache)
    hshdir = cache / hsh[0:2]
    hshpath = hshdir / hsh
    return hshpath


def load_data(hsh, verbose=False):
    """Load data for HSH.

    HSH is a string for the hash associated with the data you want.

    Returns success, data. If it succeeds, success with be True. If the data
    does not exist yet, sucess will be False, and data will be None.

    """
    hshpath = get_hashpath(hsh)
    if os.path.exists(hshpath):
        data = joblib.load(hshpath)
        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(data)
        return True, data["output"]
    else:
        return False, None


def dump_data(hsh, data, verbose):
    """Dump DATA into HSH."""
    hshpath = get_hashpath(hsh)
    os.makedirs(hshpath.parent, exist_ok=True)

    files = joblib.dump(data, hshpath)

    if verbose:
        pp = pprint.PrettyPrinter(indent=4)
        print(f"wrote {hshpath}")
        pp.pprint(data)

    return files


def hashcache(fn=None, *, verbose=False, loader=load_data, dumper=dump_data):
    """Cache results by hash of the function, arguments and kwargs.

    Set hashcache.cache to the directory you want the cache saved in.
    Default = cache
    """

    def wrapper(func, *args, **kwargs):

        hsh = get_hash(func, args, kwargs)

        # Try getting the data first
        success, data = loader(hsh, verbose)

        if success:
            return data

        # we did not succeed, so we run the function, and cache it
        # We store some metadata for future analysis.
        t0 = time.time()
        value = func(*args, **kwargs)
        tf = time.time()

        # functions with mutable arguments can change the arguments, which
        # is a problem here. We just warn the user. Nothing else makes
        # sense, the mutability may be intentional.
        if not hsh == get_hash(func, args, kwargs):
            print("WARNING something mutated, future" " calls will not use the cache.")

        # Try a bunch of ways to get a username.
        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER")

        data = {
            "output": value,
            "hash": hsh,
            "func": func.__code__.co_name,  # This is the function name
            "module": func.__module__,
            "args": args,
            "kwargs": kwargs,
            "standardized-kwargs": get_standardized_args(func, args, kwargs),
            "version": hashcache.version,
            "cwd": os.getcwd(),  # Is this a good idea? Could it leak
            # sensitive information from the path?
            # should we include other info like
            # hostname?
            "user": user,
            "run-at": t0,
            "run-at-human": time.asctime(time.localtime(t0)),
            "elapsed_time": tf - t0,
        }

        dumper(hsh, data, verbose)
        return value

    # This silliness is because I want to have the decorator work with and
    # without arguments
    #
    # @hashcache
    # def f(...)
    #
    # and
    # @hashcache(verbose=True)
    # def f(...)
    #
    # yea, it feels gross.
    if fn is not None:
        return functools.partial(wrapper, fn)
    else:

        def decorator(func):
            newrapper = functools.partial(wrapper, func)
            return functools.update_wrapper(newrapper, func)

        return decorator


hashcache.cache = "cache"
hashcache.version = "0.0.3"
