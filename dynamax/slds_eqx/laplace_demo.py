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


"""## Simulate data"""

key = jr.PRNGKey(0)

# Make a transition matrix
num_states = 5
p = (jnp.arange(num_states)**10).astype(float)
p /= p.sum()
P = jnp.zeros((num_states, num_states))
for k, p in enumerate(p[::-1]):
    P += jnp.roll(p * jnp.eye(num_states), k, axis=1)

plt.imshow(P, vmin=0, vmax=1, cmap="Greys")
plt.xlabel("next state")
plt.ylabel("current state")
plt.title("transition matrix")
plt.colorbar()

# Make dynamics distributions
latent_dim = 2
rot = lambda theta: jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                               [jnp.sin(theta), jnp.cos(theta)]])

angles = jnp.linspace(0, 2 * jnp.pi, num_states, endpoint=False)
theta = -jnp.pi / 25 # rotational frequency
As = jnp.array([0.8 * rot(theta) for _ in range(num_states)])
bs = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
Qs = jnp.tile(0.001 * jnp.eye(latent_dim), (num_states, 1, 1))

# Compute the stationary points for plotting
stationary_points = jnp.linalg.solve(jnp.eye(latent_dim) - As, bs)

# Make emission distribution
emission_dim = 10
k1, key = jr.split(key)
C = jr.normal(k1, (emission_dim, latent_dim))
d = jnp.zeros(emission_dim)
R = jnp.eye(emission_dim)

# Plot the dynamics
from dynamax.utils.plotting import COLORS as colors
from dynamax.utils.plotting import CMAP as cmap

lim = 5
x = jnp.linspace(-lim, lim, 10)
y = jnp.linspace(-lim, lim, 10)
X, Y = jnp.meshgrid(x, y)
xy = jnp.column_stack((X.ravel(), Y.ravel()))

fig, axs = plt.subplots(1, num_states, figsize=(3 * num_states, 6))
for k in range(num_states):
    A, b = As[k], bs[k]
    dxydt_m = xy.dot(A.T) + b - xy
    axs[k].quiver(xy[:, 0], xy[:, 1],
                dxydt_m[:, 0], dxydt_m[:, 1],
                color=colors[k % len(colors)])


    axs[k].set_xlabel('$x_1$')
    axs[k].set_xticks([])
    if k == 0:
        axs[k].set_ylabel("$x_2$")
    axs[k].set_yticks([])
    axs[k].set_aspect("equal")

# Pack an SLDSParams object
pi0 = jnp.ones(num_states) / num_states

log_P = tfb.SoftmaxCentered().inverse(P+1e-4)
log_Qs = tfb.Softplus(low=1e-5).inverse(vmap(jnp.diag)(Qs))
log_R = tfb.Softplus(low=1e-5).inverse(jnp.diag(R))

# Instantiate the model
slds = SLDS(num_states, latent_dim, emission_dim, log_P, As, bs, log_Qs, C, d, log_R)

print(slds.transition_logits)

# Sample the model with these params
key = jr.PRNGKey(0)
num_batches = 2
num_timesteps = 5000

zs, xs, ys = vmap(slds.sample, in_axes=(0, None))(jr.split(key, num_batches), num_timesteps)

print(zs.shape)
print(xs.shape)
print(ys.shape)

plt.figure()
plt.imshow(ys[0].T, aspect="auto", interpolation="None")
plt.colorbar()

# Plot the sampled data
fig = plt.figure(figsize=(8, 8))
for k in range(num_states):
    plt.plot(*xs[zs==k].T, 'o', color=colors[k],
         alpha=0.75, markersize=3)

plt.plot(*xs[:1000].T, '-k', lw=0.5, alpha=0.2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

def plot_states_and_timeseries(zs, vs, vs_std=None, spc=5):
    T, dim = vs.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(zs[None, :],
              extent=(0, T, -spc, dim * spc),
              cmap=cmap,
              alpha=0.5,
              aspect="auto",
              vmax=len(colors)-1)
    ax.plot(vs + spc * jnp.arange(dim), '-k', lw=1)
    if vs_std is not None:
        for d in range(dim):
            ax.fill_between(jnp.arange(T),
                            vs[:, d] + spc * d - 2 * vs_std[:, d],
                            vs[:, d] + spc * d + 2 * vs_std[:, d],
                            color='k', alpha=0.25,
                            )
    ax.set_ylim(-spc, dim * spc)
    ax.set_yticks(jnp.arange(dim) * spc)
    ax.set_yticklabels(jnp.arange(dim))
    return fig, ax

fig, ax = plot_states_and_timeseries(zs[0], xs[0])
ax.set_xlim(0, 200)
ax.set_title("latent states")

fig, ax = plot_states_and_timeseries(zs[0], ys[0])
ax.set_xlim(0, 200)
ax.set_title("emissions")

"""## Test the Laplace approximation"""

# Create the functions that we pass to laplace_approximation
P = params.transition_matrix
As = params.dynamics_matrices
bs = params.dynamics_biases
Qs = params.dynamics_covs
C = params.emission_matrix
d = params.emission_bias
R = params.emission_cov

# Define log prob functions that close over zs and params
log_prob = lambda xs, ys: slds.log_prob(ys, zs, xs, params)
initial_distribution = lambda x0: MVN(jnp.zeros(latent_dim), jnp.eye(latent_dim)).log_prob(x0)
dynamics_distribution = lambda t, xt, xtp1: MVN(As[zs[t+1]] @ xt + bs[zs[t+1]], Qs[zs[t+1]]).log_prob(xtp1)
emission_distribution = lambda t, xt, yt: MVN(C @ xt + d, R).log_prob(yt)

log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = \
    laplace_approximation(log_prob,
                        initial_distribution,
                        dynamics_distribution,
                        emission_distribution,
                        jnp.zeros_like(xs),
                        ys,
                        method="L-BFGS",
                        num_iters=50)

x_sample = block_tridiag_mvn_sample(jr.PRNGKey(0), J_diag, J_lower_diag, h)

fig, ax = plot_states_and_timeseries(zs, xs)
ax.set_xlim(0, 200)
ax.set_title("latent states")

cov_x = ExxT - jnp.einsum('ti,tj->tij', Ex, Ex)
fig, ax = plot_states_and_timeseries(zs, Ex, jnp.sqrt(vmap(jnp.diag)(cov_x)))
ax.set_xlim(0, 200)
ax.set_title("inferred latent states")

fig, ax = plot_states_and_timeseries(zs, x_sample)
ax.set_xlim(0, 200)
ax.set_title("inferred latent states")

K = num_states
D = latent_dim
N = emission_dim
T = ys.shape[0]

def _update_discrete_states(key, zs, xs, params, J_diag, J_lower_diag, h, n_discrete_samples):
    """
    Update the discrete states to the coordinate-wise maximum using the
    Viterbi algorithm.
    """
    As = params.dynamics_matrices   # (K, D, D)
    bs = params.dynamics_biases     # (K, D)
    Qs = params.dynamics_covs       # (K, D, D)

    # sample xs from q(x)
    key, *skeys = jr.split(key, n_discrete_samples+1)
    vmap_block_tridiag_mvn_sample = vmap(block_tridiag_mvn_sample, in_axes=(0, None, None, None))
    x_samples = vmap_block_tridiag_mvn_sample(jnp.array(skeys), J_diag, J_lower_diag, h)

    # TODO: replace with initial state distribution object
    initial_state_distn = jnp.ones(K) / K
    pi0 = jnp.mean(jnp.array(
        [initial_state_distn
            for x in x_samples]), axis=0)

    # TODO: eventually, transition matrix will depend on x
    # this should be another to call to a function that returns transition matrices
    P = jnp.mean(jnp.array(
        [params.transition_matrix
            for x in x_samples]), axis=0)

    def _dynamics_likelihood(xs):
        means = jnp.einsum('kde,te->tkd', As, xs[:-1]) + bs
        log_likes = MVN(means, Qs).log_prob(xs[1:][:, None, :]) # (T-1, K)
        # Account for the first timestep
        log_likes = jnp.vstack([
            MVN(jnp.zeros(D), jnp.eye(D)).log_prob(xs[0]) * jnp.ones((1, K)),
            log_likes])
        return log_likes
    vmap_dynamics_likelihood = vmap(_dynamics_likelihood)

    log_likes = jnp.mean(vmap_dynamics_likelihood(x_samples), axis=0)

    return hmm.inference.hmm_smoother(pi0, P, log_likes)

n_discrete_samples = 1
post = _update_discrete_states(jr.PRNGKey(0), 0, 0, params, J_diag, J_lower_diag, h, n_discrete_samples)
zs_lem = jnp.argmax(post.smoothed_probs, axis=1)

fig, ax = plot_states_and_timeseries(zs, xs)
ax.set_xlim(0, 200)
ax.set_title("latent states")

cov_x = ExxT - jnp.einsum('ti,tj->tij', Ex, Ex)
fig, ax = plot_states_and_timeseries(zs, Ex, jnp.sqrt(vmap(jnp.diag)(cov_x)))
ax.set_xlim(0, 200)
ax.set_title("inferred latent states")

fig, ax = plot_states_and_timeseries(zs_lem, Ex)
ax.set_xlim(0, 200)
ax.set_title("inferred discrete states & continuous mean")

"""## Fit with Laplace EM"""

# test_slds = SLDS(num_states, latent_dim, emission_dim, pi0, P, As, bs, Qs, C, d, R)


fit_slds, lps, fit_zs, fit_xs, final_key = fit_laplace_em(slds,
    jr.PRNGKey(0), ys, zs, xs, num_iters=10, n_discrete_samples=1)

assert jnp.all(jnp.isfinite(lps))

vmap(fit_slds.log_prob)(ys, fit_zs, fit_xs).sum()

def _objective(slds):
    return -1.0 * jnp.sum(vmap(slds.log_prob)(ys, zs, xs)) / ys.size

g = grad(_objective)(slds)

optimizer = optax.adam(1.0)
opt_state = optimizer.init(fit_slds)
updates, opt_state = optimizer.update(g, opt_state)
new_slds = optax.apply_updates(fit_slds, updates)

# opt_state = optim.init(eqx.filter(model, eqx.is_array))

print(fit_slds.transition_logits)
print(slds.transition_logits)

plt.plot(lps)
plt.xlabel("iteration")
plt.ylabel("log prob")

tr = 0

fig, ax = plot_states_and_timeseries(fit_zs[tr], fit_xs[tr])
ax.set_xlim(0, 200)
ax.set_title("fitted latent states")

fig, ax = plot_states_and_timeseries(zs[tr], xs[tr])
ax.set_xlim(0, 200)
ax.set_title("true latent states")

"""## Fit the model to data (starting from the true parameters)"""

slds = SLDS(num_states, latent_dim, emission_dim)
lps, fit_zs, fit_xs, fit_params = slds.fit(ys, zs, xs, params, num_iters=5)
assert jnp.all(jnp.isfinite(lps))

plt.plot(lps)
plt.xlabel("iteration")
plt.ylabel("log prob")

fig, ax = plot_states_and_timeseries(fit_zs, fit_xs)
ax.set_xlim(0, 200)
ax.set_title("fitted latent states")

fig, ax = plot_states_and_timeseries(zs, xs)
ax.set_xlim(0, 200)
ax.set_title("true latent states")

# Plot the sampled data
fig = plt.figure(figsize=(8, 8))
for k in range(num_states):
    plt.plot(*fit_xs[fit_zs==k].T, 'o', color=colors[k],
         alpha=0.75, markersize=3)

plt.plot(*fit_xs[:1000].T, '-k', lw=0.5, alpha=0.2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("fitted latent states")

params.emission_matrix

fit_params.emission_matrix

"""## Now fit the model from a random initialization"""

