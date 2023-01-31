"""hashcache - a decorator for persistent, file/hash-based cache

I found some features of joblib were unsuitable for how I want to use a cache.

1. The "file" Python things the function is in is used to save the results in
joblib, which leads to repeated runs if you run the same code in Python,
notebook or stdin, and means the cache is not portable to other machines, and
maybe not even in time since temp directories and kernel parameters are
involved. I could not figure out how to change those in joblib.

2. joblib uses the function source code in the hash, so inconsequential changes
like whitespace, docstrings and comments change the hash.

This library aims to provide a simpler version of what I wish joblib did for me.

Results are cached based on a hash of the function name, argnames, bytecode, arg
values and kwarg values. I use joblib.hash for this. This means any two
functions with the same bytecode, even if they have different names,

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

This is alpha, proof of concept code. Test it a lot for your use case.

"""

import functools
import inspect
import joblib
import os
from pathlib import Path
import pprint
import time


def get_hash(func, args, kwargs):
    g = dict(inspect.getmembers(func))["__globals__"]

    return joblib.hash(
        [
            func.__code__.co_name,  # This is the function name
            func.__code__.co_varnames,  # names of arguments,
            func.__code__.co_names,  # these are other things than
            # args, usually outside vars
            # I don't know how to get their values though
            [g[var] for var in func.__code__.co_names],
            func.__code__.co_code,
            args,
            kwargs,
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

        # Note I used to try getting the source of the function with inspect.
        # That worked well in notebooks, but not in Python scripts for some
        # reason. Here I use attributes of the code objects. I include the
        # function name, variable names, names of other variables, the function
        # bytecode, and args and kwargs values. I do not know how portable this
        # is, e.g. to other machines, Python upgrades, etc. The reason I use all
        # these is to avoid hash conflicts from functions with the same body
        # (which have the same bytecode) and to make sure external variable
        # changes trigger new runs.
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
            g = dict(inspect.getmembers(func))["__globals__"]
            data = {
                "output": value,
                "args": args,
                "arg-names": func.__code__.co_names,
                "other-names": func.__code__.co_names,
                "other-values": [g[var] for var in func.__code__.co_names],
                "kwargs": kwargs,
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
hashcache.version = "0.0.1"
