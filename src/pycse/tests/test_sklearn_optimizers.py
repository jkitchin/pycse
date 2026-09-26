"""Tests for pycse.sklearn.optimizers.run_optimizer name handling."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from pycse.sklearn.optimizers import run_lbfgs, run_optimizer


def _loss(p):
    return jnp.sum((p - 1.0) ** 2)


P0 = jnp.zeros(3)


@pytest.mark.parametrize("name", ["lbfgsb", "nonlinear_cg", "LBFGSB"])
def test_legacy_names_warn_and_map_to_lbfgs(name):
    with pytest.warns(UserWarning, match="not implemented as a distinct algorithm"):
        params, state = run_optimizer(name, _loss, P0, maxiter=50)
    ref_params, ref_state = run_lbfgs(_loss, P0, maxiter=50)
    np.testing.assert_allclose(params, ref_params)
    assert state.iter_num == ref_state.iter_num


@pytest.mark.parametrize("name", ["lbfgs", "bfgs", "adam", "adam_cosine"])
def test_supported_names_do_not_warn(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        params, state = run_optimizer(name, _loss, P0, maxiter=20)
    assert np.all(np.isfinite(params))


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        run_optimizer("newton", _loss, P0, maxiter=5)


def _rosenbrock(p):
    return jnp.sum(100.0 * (p[1:] - p[:-1] ** 2) ** 2 + (1.0 - p[:-1]) ** 2)


@pytest.mark.parametrize("name", ["lbfgs", "bfgs"])
def test_lbfgs_is_quasi_newton(name):
    """L-BFGS solves Rosenbrock in far fewer iterations than a first-order method."""
    params, state = run_optimizer(name, _rosenbrock, jnp.zeros(4), maxiter=200, tol=1e-4)
    np.testing.assert_allclose(params, np.ones(4), atol=1e-3)
    assert state.converged
    assert state.iter_num < 100
    assert state.value < 1e-6


def test_lbfgs_pytree_params():
    params = {"a": jnp.zeros(2), "b": jnp.zeros((2, 2))}

    def loss(p):
        return jnp.sum((p["a"] - 2.0) ** 2) + jnp.sum((p["b"] + 1.0) ** 2)

    out, state = run_lbfgs(loss, params, maxiter=50, tol=1e-6)
    np.testing.assert_allclose(out["a"], 2.0, atol=1e-5)
    np.testing.assert_allclose(out["b"], -1.0, atol=1e-5)
    assert state.converged


def test_lbfgs_stops_before_non_finite():
    """If the loss becomes non-finite, the last finite parameters are returned."""

    def loss(p):
        # finite only for p < 2; the unconstrained minimum at p = 3 is unreachable
        return jnp.where(p[0] < 2.0, (p[0] - 3.0) ** 2, jnp.nan)

    params, state = run_lbfgs(loss, jnp.zeros(1), maxiter=50)
    assert np.all(np.isfinite(params))
    assert np.isfinite(state.value)


def test_adam_cosine_is_old_schedule():
    params, state = run_optimizer("adam_cosine", _loss, P0, maxiter=300)
    assert np.all(np.isfinite(params))
    assert state.value < _loss(P0)
