"""Tests for the lazy exports of pycse.sklearn."""

import importlib

import pytest

import pycse.sklearn as pkg

# Optional dependencies that some estimators need; names depending on a missing
# optional package are skipped rather than failed.
# gmr is a declared dependency, but test_sklearn_nngmm also skips without it.
OPTIONAL_DEPS = {"torch", "nflows", "TorchSisso", "gmr"}


def _import_or_skip(module):
    """Import a module, skipping the test if an optional dependency is missing."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in OPTIONAL_DEPS:
            pytest.skip(f"optional dependency {e.name!r} not installed")
        raise


@pytest.mark.parametrize("name", pkg.__all__)
def test_all_names_resolve(name):
    """Every name in __all__ resolves to a class via lazy import."""
    try:
        obj = getattr(pkg, name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in OPTIONAL_DEPS:
            pytest.skip(f"optional dependency {e.name!r} not installed")
        raise
    assert isinstance(obj, type)


@pytest.mark.parametrize(
    "alias, module, cls",
    [
        ("LLPR", "pycse.sklearn.llpr_regressor", "LLPRRegressor"),
        ("NNBR", "pycse.sklearn.nnbr", "NeuralNetworkBLR"),
        ("NNGMM", "pycse.sklearn.nngmm", "NeuralNetworkGMM"),
    ],
)
def test_short_aliases_match_real_classes(alias, module, cls):
    """Short aliases and full class names refer to the same class."""
    real = getattr(_import_or_skip(module), cls)
    assert getattr(pkg, alias) is real
    assert getattr(pkg, cls) is real


def test_from_import_aliases():
    """`from pycse.sklearn import X` works for the aliases."""
    _import_or_skip("pycse.sklearn.nngmm")
    from pycse.sklearn import LLPR, NNBR, NNGMM, LinearRegressionUQ  # noqa: F401


def test_unknown_name_raises_attribute_error():
    """Unknown names raise AttributeError (not ImportError)."""
    with pytest.raises(AttributeError):
        pkg.DoesNotExist  # noqa: B018
    assert not hasattr(pkg, "DoesNotExist")


def test_dir_lists_lazy_names():
    """dir() includes the lazily exported names."""
    assert set(pkg.__all__) <= set(dir(pkg))
