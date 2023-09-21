"""A supervisor decorator to rerun functions if they can be fixed.

Functions can work but fail, or fail by raising exceptions. If you can examine
the ouput of a function and algorithmically propose a new set of arguments to
fix it, then supervisor can help you automate this. The idea is to write check
and exception functions for this, and then decorate your function to use them.

This is an alternative to Custodian
(http://materialsproject.github.io/custodian/) which is pretty awesome, but also
heavy-weight IMO.

This library was a proof of concept in making a decorator for this purpose. In
the end, I am not sure it is less heavyweight than custodian.

[2023-09-20 Wed] Lightly tested on some examples.
[2023-09-21 Thu] Added the manager.
"""

import functools
import inspect


def supervisor(check_funcs=(), exception_funcs=(), max_errors=5, verbose=False):
    """Decorator to supervise a function. After the function is run, each
    function in CHECK_FUNCS is run on the result. Each checker function has the
    signature check(args, kwargs, result). If the function should be rerun, then
    the check function should return a new args, kwargs to rerun the function
    with. Otherwise, it should return None

    If there is an exception in the function, then each function in
    EXCEPTION_FUNCS will be run. Each exception function has the signature
    func(args, kwargs, exc). If one of them can fix the issue, it should return
    a new (args, kwargs) to rerun the function with, and otherwise return None
    which indicates there is no fix.

    MAX_ERRORS is the maximum number of issues to try to fix. A value of -1
    means try forever.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nerrors = 0
            run = args, kwargs

            while run and (nerrors < max_errors):
                try:
                    result = func(*run[0], **run[1])
                    for checker in check_funcs:
                        # run is None if everything checks out
                        # or run is (args, kwargs) if it needs to be run again
                        args, kwargs = run
                        run = checker(args, kwargs, result)
                        if run:
                            if verbose:
                                s = getattr(checker, "__name__", checker)
                                print(f"Proposed fix in {s}: {run}")
                            nerrors += 1
                            # short-circuit break because we need to run it now.
                            # this is a sequential fix, and does not allow a way
                            # to choose what to fix if there is more than one
                            # error
                            break
                    # After all the checks, run is None if they all passed, that
                    # means we should return
                    if not check_funcs or run is None:
                        return result
                    # Now should be returning to the while loop with new params
                    # in run
                except Exception as e:
                    if not exception_funcs:
                        raise (e)  # no fixer funcs defined, so we re-raise

                    for exc in exception_funcs:
                        run = exc(run[0], run[1], e)
                        if run:
                            if verbose:
                                s = getattr(exc, "__name__", exc)
                                print(f"Proposed fix in {s}: {run}")
                            nerrors += 1
                            break  # break out as soon as we get a fix

                    if run is None:
                        # no new thing to try, reraise
                        raise (e)

                    # if run is not None, this goes back to the while loop with
                    # new params in run

            # after the loop, we should raise if we got too many errors
            if nerrors == max_errors:
                raise Exception("Too many errors found")

        return wrapper

    return decorator


# Version 2


def check_result(func):
    """Decorator for functions to check the function result."""

    def wrapper(arguments, result):
        if isinstance(result, Exception):
            return None
        else:
            return func(arguments, result)

    return wrapper


def check_exception(func):
    """Decorator for functions to fix exceptions."""

    def wrapper(arguments, result):
        if isinstance(result, Exception):
            return func(arguments, result)
        else:
            return None

    return wrapper


def manager(checkers=(), max_errors=5, verbose=False):
    """Decorator to manage a function. After the function is run, each function
    in CHECKERS is run on the result. Each checker function has the signature
    check(arguments, result). arguments will always be a dictionary of kwargs,
    including the default values. If the function should be rerun, then the
    checker function should return a new arguments dictionary to rerun the
    function with. Otherwise, it should return None.

    The checker functions should be decorated with check_results or
    check_exception to indicate which one they handle.

    MAX_ERRORS is the maximum number of issues to try to fix. A value of -1
    means try forever.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nerrors = 0

            # build the kwargs representation
            # this converts args to kwargs
            sig = inspect.signature(func)
            normalized_args = sig.bind(*args, **kwargs)
            normalized_args.apply_defaults()
            runargs = normalized_args.arguments

            while runargs and (nerrors < max_errors):
                try:
                    result = func(**runargs)
                    for checker in checkers:
                        # run is None if everything checks out
                        # or run is (args, kwargs) if it needs to be run again
                        rerun_args = checker(runargs, result)
                        if rerun_args:
                            runargs = rerun_args
                            if verbose:
                                s = getattr(checker, "__name__", checker)
                                print(f"Proposed fix in {s}: {runargs}")
                            nerrors += 1
                            # short-circuit break because we need to run it now.
                            # this is a sequential fix, and does not allow a way
                            # to choose what to fix if there is more than one
                            # error
                            break
                    # After all the checks, run is None if they all passed, that
                    # means we should return
                    if not checkers or rerun_args is None:
                        return result
                    # Now should be returning to the while loop with new params
                    # in run

                except Exception as e:
                    for checker in checkers:
                        rerun_args = checker(runargs, e)
                        if rerun_args:
                            runargs = rerun_args
                            if verbose:
                                s = getattr(checker, "__name__", checker)
                                print(f"Proposed fix in {s}: {runargs}")
                            nerrors += 1
                            break  # break out as soon as we get a fix

                    if rerun_args is None:
                        # no new arguments to rerun with were found
                        # so nothing can be fixed.
                        raise (e)

                    # if runargs is not None, this goes back to the while loop
                    # with new params in run

            # after the loop, we should raise if we got too many errors
            if nerrors == max_errors:
                raise Exception("Too many errors found")

        return wrapper

    return decorator
