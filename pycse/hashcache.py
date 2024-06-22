"""hashcache - a class decorator for persistent, file/hash-based cache

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

The cache location is set as a class attribute:

    HashCache.cache = './cache'


    HashCache - stores joblib.dump pickle strings in files named by hash


    SqlCache - stores orjson serialized data in a sqlite3 database by hash key

    JsonCache - stores orjson serialized data in json files, compatible with maggma


This is still alpha, proof of concept code. Test it a lot for your use case. The
API is not stable, and subject to change.


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

[2024-06-18 Tue] Changed from function to class decorator (breaking change).

"""

import inspect
import joblib
import orjson
import os
from pathlib import Path
import pprint
import socket
import sqlite3
import time


def hashcache(*args, **kwargs):
    """Raises an exception if the old hashcache decorator is used."""
    raise Exception(
        "The hashcache function decorator is deprecated." " Please use the class decorator instead."
    )


class HashCache:
    """Class decorator to cache using hashes and pickle (via joblib).
    Data is stored in directories named by the hash.

    """

    # cache is the name of the directory to store results in
    cache = "cache"
    version = "0.1.0"
    verbose = False

    def __init__(self, function):
        self.function = function

    def get_standardized_args(self, args, kwargs):
        """Returns a standardized dictionary of kwargs for func(args, kwargs)

        This dictionary includes default values, even if they were not called.

        """
        sig = inspect.signature(self.function)
        standardized_args = sig.bind(*args, **kwargs)
        standardized_args.apply_defaults()
        return standardized_args.arguments

    def get_hash(self, args, kwargs):
        """Get a hash for running FUNC(ARGS, KWARGS).

        This is the most critical feature of hashcache as it provides a key to store
        and look up results later. You should think carefully before changing this
        function, it breaks past caches.

        FUNC should be as pure as reasonable. This hash is insensitive to global
        variables.

        The hash is on the function name, bytecode, and a standardized kwargs
        including defaults. We use bytecode because it is insensitive to things
        like whitespace, comments, docstrings, and variable name changes that
        don't affect results. It is assumed that two functions with the same
        name and bytecode will evaluate to the same result. However, this makes
        the hash fragile to changes in Python version that affect bytecode.

        """
        return joblib.hash(
            [
                self.function.__code__.co_name,  # This is the function name
                self.function.__code__.co_code,  # this is the function bytecode
                self.get_standardized_args(args, kwargs),  # The args used, including defaults
            ],
            hash_name="sha1",
        )

    def get_hashpath(self, hsh):
        """Return path to file for HSH."""
        cache = Path(self.cache)
        hshdir = cache / hsh[0:2]
        hshpath = hshdir / hsh
        return hshpath

    def load_data(self, hsh):
        """Load data for HSH.

        HSH is a string for the hash associated with the data you want.

        Returns success, data. If it succeeds, success with be True. If the data
        does not exist yet, sucess will be False, and data will be None.

        """
        hshpath = self.get_hashpath(hsh)
        if os.path.exists(hshpath):
            data = joblib.load(hshpath)
            if self.verbose:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(data)
            return True, data["output"]
        else:
            return False, None

    def dump_data(self, hsh, data):
        """Dump DATA into HSH."""
        hshpath = self.get_hashpath(hsh)
        os.makedirs(hshpath.parent, exist_ok=True)

        files = joblib.dump(data, hshpath)

        if self.verbose:
            pp = pprint.PrettyPrinter(indent=4)
            print(f"wrote {hshpath}")
            pp.pprint(data)

        return files

    def __call__(self, *args, **kwargs):
        """This is the decorator code that runs around self.function."""

        hsh = self.get_hash(args, kwargs)

        # Try getting the data first
        success, data = self.load_data(hsh)

        if success:
            return data

        # we did not succeed, so we run the function, and cache it
        # We store some metadata for future analysis.
        t0 = time.time()
        value = self.function(*args, **kwargs)
        tf = time.time()

        # functions with mutable arguments can change the arguments, which
        # is a problem here. We just warn the user. Nothing else makes
        # sense, the mutability may be intentional.
        if not hsh == self.get_hash(args, kwargs):
            print("WARNING something mutated, future" " calls will not use the cache.")

        # Try a bunch of ways to get a username.
        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER")

        data = {
            "output": value,
            "hash": hsh,
            "func": self.function.__code__.co_name,  # This is the function name
            "module": self.function.__module__,
            "args": args,
            "kwargs": kwargs,
            "standardized-kwargs": self.get_standardized_args(args, kwargs),
            "version": self.version,
            "cwd": os.getcwd(),
            "hostname": socket.getfqdn(),
            "user": user,
            "run-at": t0,
            "run-at-human": time.asctime(time.localtime(t0)),
            "elapsed_time": tf - t0,
        }

        self.dump_data(hsh, data)
        return value

    @staticmethod
    def dump(**kwargs):
        """Dump KWARGS to the cache.
        Returns a hash string for future lookup.

        cache is a special kwarg that is not saved

        """
        t0 = time.time()
        hsh = joblib.hash(kwargs)

        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER")

        if "cache" in kwargs:
            cache = kwargs["cache"]
            del kwargs["cache"]
        else:
            cache = "cache"

        data = {
            "func": "dump",
            "kwargs": kwargs,
            "hash": hsh,
            "saved-at": t0,
            "saved-at-human": time.asctime(time.localtime(t0)),
            "cwd": os.getcwd(),
            "hostname": socket.getfqdn(),
            "user": user,
        }

        hc = HashCache(lambda x: x)
        hc.cache = cache
        hc.dump_data(hsh, data)
        return hsh

    @staticmethod
    def load(hsh, cache="cache"):
        """Load saved variables from HSH."""
        hc = HashCache(lambda x: x)
        hc.cache = cache

        hshpath = hc.get_hashpath(hsh)
        if os.path.exists(hshpath):
            return joblib.load(hshpath)["kwargs"]


class SqlCache(HashCache):
    """Class decorator to cache using orjson and sqlite.
    Data is stored in a sqlite database as json.

    """

    cache = "cache.sqlite"

    def __init__(self, function):
        self.function = function

        self.con = sqlite3.connect(self.cache)
        self.con.execute("CREATE TABLE if not exists cache(hash TEXT unique, value TEXT)")

    def dump_data(self, hsh, data):
        """Dump DATA into HSH.
        DATA must be serializable to json.

        """
        value = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
        with self.con:
            self.con.execute("INSERT INTO cache(hash, value) VALUES(?, ?)", (hsh, value))

    def load_data(self, hsh):
        """Load data for HSH.

        HSH is a string for the hash associated with the data you want.

        Returns success, data. If it succeeds, success with be True. If the data
        does not exist yet, sucess will be False, and data will be None.

        """
        with self.con:
            cur = self.con.execute("SELECT value FROM cache WHERE hash = ?", (hsh,))
            value = cur.fetchone()
        if value is None:
            return False, None
        else:
            return True, orjson.loads(value[0])["output"]

    @staticmethod
    def search(query, *args):
        """Run a sql QUERY with args.
        args are substituted in ? placeholders in the query.

        This is just a light wrapper on con.execute.

        """
        con = sqlite3.connect(SqlCache.cache)
        cur = con.execute(query, args)
        return cur

    @staticmethod
    def dump(**kwargs):
        """Dump KWARGS to the cache.
        Returns a hash string for future lookup.
        """
        t0 = time.time()
        hsh = joblib.hash(kwargs)

        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER")

        data = {
            "func": "dump",
            "kwargs": kwargs,
            "hash": hsh,
            "saved-at": t0,
            "saved-at-human": time.asctime(time.localtime(t0)),
            "cwd": os.getcwd(),
            "hostname": socket.getfqdn(),
            "user": user,
        }

        hc = SqlCache(lambda x: x)
        try:
            hc.dump_data(hsh, data)
            return hsh
        except sqlite3.IntegrityError:
            return hsh

    @staticmethod
    def load(hsh):
        """Load data from HSH."""

        hc = SqlCache(lambda x: x)
        with hc.con:
            cur = hc.con.execute("SELECT value FROM cache WHERE hash = ?", (hsh,))
            (value,) = cur.fetchone()  # this returns a tuple that we unpack
            return orjson.loads(value)["kwargs"]


class JsonCache(HashCache):
    """Json-based cache.

    This is compatible with maggma.
    """

    def __init__(self, function):
        self.function = function

        if not os.path.exists(self.cache / Path("Filestore.json")):
            os.makedirs(self.cache, exist_ok=True)
            with open(self.cache / Path("Filestore.json"), "wb") as f:
                f.write(orjson.dumps([]))

    def dump_data(self, hsh, data):
        """Dump DATA into HSH."""
        hshpath = self.get_hashpath(hsh).with_suffix(".json")
        os.makedirs(hshpath.parent, exist_ok=True)

        with open(hshpath, "wb") as f:
            f.write(orjson.dumps(data))

    def load_data(self, hsh):
        hshpath = self.get_hashpath(hsh).with_suffix(".json")
        if os.path.exists(hshpath):
            with open(hshpath, "rb") as f:
                data = orjson.loads(f.read())

            if self.verbose:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(data)
            return True, data["output"]
        else:
            return False, None

    @staticmethod
    def dump(**kwargs):
        """Dump KWARGS to the cache.
        Returns a hash string for future lookup.
        """
        t0 = time.time()
        hsh = joblib.hash(kwargs)

        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER")

        data = {
            "func": "dump",
            "kwargs": kwargs,
            "hash": hsh,
            "saved-at": t0,
            "saved-at-human": time.asctime(time.localtime(t0)),
            "cwd": os.getcwd(),
            "hostname": socket.getfqdn(),
            "user": user,
        }

        hc = JsonCache(lambda x: x)
        hshpath = hc.get_hashpath(hsh).with_suffix(".json")

        os.makedirs(hshpath.parent, exist_ok=True)
        with open(hshpath, "wb") as f:
            f.write(orjson.dumps(data))
        return hsh

    @staticmethod
    def load(hsh):
        """Load data from HSH."""

        hc = JsonCache(lambda x: x)
        hshpath = hc.get_hashpath(hsh).with_suffix(".json")
        if os.path.exists(hshpath):
            with open(hshpath, "rb") as f:
                return orjson.loads(f.read())["kwargs"]
