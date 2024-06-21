import dataclasses
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns

from jax import grad, jit, vmap, lax
from jax.nn import one_hot
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Float, PyTree
from typing import Callable
from tqdm.auto import trange
from tensorflow_probability.substrates import jax as tfp

import equinox as eqx

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm
from dynamax import linear_gaussian_ssm as lds
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params

import optax
import jax.scipy.optimize
from jax import hessian, value_and_grad, jacfwd, jacrev
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations


from jax.flatten_util import ravel_pytree
from functools import partial

from jax.scipy.special import logsumexp

# Helper functions

# for fitting a linear regression given sufficient stats
def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), b)
    return x


def fit_linear_regression(ExxT, ExyT, EyyT, N, shrinkage=1e-4):
    # Solve a linear regression given sufficient statistics
    D_in, D_out = ExyT.shape[-2:]
    W = psd_solve(ExxT + 1e-4 * jnp.eye(D_in), ExyT).T
    Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N + shrinkage * jnp.eye(D_out)
    return W, symmetrize(Sigma)

def convex_combo(x, y, step_size):
    return  (1.0 - step_size) * x + step_size * y

"""## Generic code for Laplace"""

from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_log_normalizer

def _sample_info_gaussian(key, J, h, sample_shape=()):
    # TODO: avoid inversion.
    # see https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py#L117-L122
    # L = np.linalg.cholesky(J)
    # x = jr.normal(key, h.shape[0])
    # return scipy.linalg.solve_triangular(L,x,lower=True,trans='T') \
    #     + dpotrs(L,h,lower=True)[0]
    cov = jnp.linalg.inv(J)
    loc = jnp.einsum("...ij,...j->...i", cov, h)
    return tfp.distributions.MultivariateNormalFullCovariance(
        loc=loc, covariance_matrix=cov).sample(sample_shape=sample_shape, seed=key)


def block_tridiag_mvn_sample(key, J_diag, J_lower_diag, h):
    # Run the forward filter
    log_Z, (filtered_Js, filtered_hs) = block_tridiag_mvn_log_normalizer(J_diag, J_lower_diag, h)

    # Backward sample
    def _step(carry, inpt):
        x_next, key = carry
        Jf, hf, L = inpt

        # Condition on the next observation
        Jc = Jf
        # hc = hf - jnp.einsum('ni,ij->nj', x_next, L)
        hc = hf - L.T @ x_next

        # Split the key
        key, this_key = jr.split(key)
        x = _sample_info_gaussian(this_key, Jc, hc)
        return (x, key), x

    # Initialize with sample of last timestep and sample in reverse
    last_key, key = jr.split(key)
    x_T = _sample_info_gaussian(last_key, filtered_Js[-1], filtered_hs[-1])

    # inputs = (filtered_Js[:-1][::-1], filtered_hs[:-1][::-1], J_lower_diag[::-1])
    # _, x_rev = lax.scan(_step, (x_T, key), inputs)
    _, x = lax.scan(_step, (x_T, key), (filtered_Js[:-1], filtered_hs[:-1], J_lower_diag), reverse=True)

    # Reverse and concatenate the last time-step's sample
    x = jnp.concatenate((x, x_T[None, ...]), axis=0)

    # Transpose to be (num_samples, num_timesteps, dim)
    return x
