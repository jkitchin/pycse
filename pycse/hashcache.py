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

    hashcache.cachedir = './cache'

Caveats:

1. hash-based approaches are fragile. If the function bytecode changes, the hash
will change. That means the cache may not work across Python versions, or if
some kind of optimization is used, etc.

There is no utility for pruning the cache. You can simply delete the directory,
or create a new one.

You can use hashcache.dryrun to avoid reading/writing and just return the cache.
Here is a way to delete a cache entry.

    hashcache.dryrun = True

    h = f(22)

    if os.path.exists(h): os.unlink(h)

This is alpha, proof of concept code. Test it a lot for your use case. The API
is not stable, and subject to change.

Pros:

1. File-based cache which means many functions can run in parallel reading and
writing, and you are limited only by file io speeds, and disk space.

2. semi-portability. The cachedir could be synced across machines.

3. No server.

Cons:

1. File-based cache which means if you generate thousands of files, it can be
slow to delete them. Although it should be fast to access the results, it will
not be fast to iterate over all the results, e.g. if you want to implement some
kind of search or reporting.

[2023-09-23 Sat] Changed hash signature (breaking change)

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

    # We get all the arguments, including defaults, and standardize them for the
    # hash.

    return joblib.hash(
        [
            func.__code__.co_name,  # This is the function name
            func.__code__.co_code,  # this is the function bytecode
            get_standardized_args(
                func, args, kwargs
            ),  # The args used, including defaults
        ],
        hash_name="sha1",
    )


def hashcache(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        """Cache results by hash of the function source, arguments and kwargs.

        Set hashcache.cachedir to the directory you want the cache saved in.
        Default = ./cache Set hashcache.verbose to True to get more verbosity.
        """

        hsh = get_hash(func, args, kwargs)

        cache = Path(hashcache.cachedir)
        hshdir = cache / hsh[0:2]
        hshpath = hshdir / hsh

        if hashcache.dryrun:
            if hashcache.verbose:
                print(f"Dry run in {hshpath}")
            return hshpath

        if hashcache.delete:
            if os.path.exists(hshpath):
                if hashcache.verbose:
                    print(f"Deleting {hshpath}")
                return os.unlink(hshpath)
            else:
                print(f"{hshpath} not found")
                return None

        # If the hshpath exists, we can read from it.
        if os.path.exists(hshpath):
            if hashcache.verbose:
                print(f"Reading from {hshpath}")

            data = joblib.load(hshpath)
            if hashcache.verbose:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(data)
            return data["output"]
        # It didn't exist, so we run the function, and cache it
        else:
            t0 = time.time()
            value = func(*args, **kwargs)
            tf = time.time()

            # functions with mutable arguments can change the arguments, which
            # is a problem here. We just warn the user. Nothing else makes
            # sense, the mutability may be intentional.
            if not hsh == get_hash(func, args, kwargs):
                print(
                    "WARNING something mutated, future"
                    " calls will not use the cache."
                )

            os.makedirs(hshdir, exist_ok=True)
            data = {
                "output": value,
                "func": func.__code__.co_name,  # This is the function name
                "module": func.__module__,
                "args": args,
                "kwargs": kwargs,
                "standardized-kwargs": get_standardized_args(
                    func, args, kwargs
                ),
                "cwd": os.getcwd(),  # Is this a good idea? Could it leak
                # sensitive information from the path?
                # should we include other info like
                # hostname?
                "user": os.getlogin(),
                "run-at": t0,
                "run-at-human": time.asctime(time.localtime(t0)),
                "elapsed_time": tf - t0,
                "version": hashcache.version,
            }

            joblib.dump(data, hshpath)
            if hashcache.verbose:
                pp = pprint.PrettyPrinter(indent=4)
                print(f"wrote {hshpath}")
                pp.pprint(data)

            return value

    return wrapper_decorator


hashcache.cachedir = "./cache"
hashcache.dryrun = False
hashcache.delete = False
hashcache.verbose = False
hashcache.version = "0.0.2"
