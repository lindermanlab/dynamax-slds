import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, vmap, lax
from tensorflow_probability.substrates import jax as tfp

import equinox as eqx

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm

import optax
import jax.scipy.optimize
from jax import hessian, value_and_grad, jacfwd, jacrev
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations
from functools import partial



def laplace_approximation(log_prob,
                          initial_distribution,
                          dynamics_distribution,
                          emission_distribution,
                          initial_states,
                          emissions,
                          method="BFGS",
                          adam_learning_rate=1e-2,
                          num_iters=10):
    """
    Laplace approximation to the posterior distribution for state space models
    with continuous latent states.

    log_prob: states, emissions -> log prob (scalar)
    initial_distribution: initial state -> log prob (scalar)
    dynamics_distribution: time, curr_state, next_state -> log prob (scalar)
    emission_distribution: time, curr_state, curr_emission -> log prob (scalar)
    x0 (array, (num_timesteps, latent_dim)): Initial guess of state mode.
    data (array, (num_timesteps, obs_dim)): Observation data.
    method (str, optional): Optimization method to use. Choices are
        ["L-BFGS", "BFGS", "Adam"]. Defaults to "L-BFGS".
    learning_rate (float, optional): [description]. Defaults to 1e-3.
    num_iters (int, optional): Only used when optimization method is "Adam."
        Specifies the number of update iterations. Defaults to 50.

    """
    def _compute_laplace_mean(initial_states):
        """Find the mode of the log joint probability for the Laplace approximation.
        """

        scale = initial_states.size
        dim = initial_states.shape[-1]

        if method == "BFGS" or "L-BFGS":
            # scipy minimize expects x to be shape (n,) so we flatten / unflatten
            def _objective(x_flattened):
                x = x_flattened.reshape(-1, dim)
                return -1 * jnp.sum(log_prob(x, emissions)) / scale

            optimize_results = jax.scipy.optimize.minimize(
                _objective,
                initial_states.ravel(),
                method="bfgs" if method == "BFGS" else "l-bfgs-experimental-do-not-rely-on-this",
                options=dict(maxiter=num_iters))

            # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
            x_mode = optimize_results.x.reshape(-1, dim)  # reshape back to (T, D)

        elif method == "Adam":

            params = initial_states
            _objective = lambda x: -1 * jnp.sum(log_prob(x, emissions)) / scale
            optimizer = optax.adam(adam_learning_rate)
            opt_state = optimizer.init(params)

            @jit
            def step(params, opt_state):
                loss_value, grads = value_and_grad(_objective)()
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss_value

            # TODO: Replace with a scan
            for i in range(num_iters):
                params, opt_state, loss_value = step(params, opt_state)
            x_mode = params

        else:
            raise ValueError(f"method = {method} is not recognized. Should be one of ['Adam', 'BFGS']")

        return x_mode

    def _compute_laplace_precision_blocks(states):
        """Get the negative Hessian at the given states for the Laplace approximation.
        """
        # initial distribution
        J_init = -1 * hessian(initial_distribution)(states[0])

        # dynamics
        f = dynamics_distribution
        ts = jnp.arange(len(states))
        J_11 = -1 * vmap(hessian(f, argnums=1))(ts[:-1], states[:-1], states[1:])
        J_22 = -1 * vmap(hessian(f, argnums=2))(ts[:-1], states[:-1], states[1:])
        J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=2), argnums=1))(ts[:-1], states[:-1], states[1:])

        # emissions
        f = emission_distribution
        J_obs = -1 * vmap(hessian(f, argnums=1))(ts, states, emissions)

        # debug only if this flag is set
        # if jax.config.jax_disable_jit:
        #     assert not np.any(np.isnan(J_init)), "nans in J_init"
        #     assert not np.any(np.isnan(J_11)), "nans in J_11"
        #     assert not np.any(np.isnan(J_22)), "nans in J_22"
        #     assert not np.any(np.isnan(J_21)), "nans in J_21"
        #     assert not np.any(np.isnan(J_obs)), "nans in J_obs"

        # combine into diagonal and lower diagonal blocks
        J_diag = J_obs
        J_diag = J_diag.at[0].add(J_init)
        J_diag = J_diag.at[:-1].add(J_11)
        J_diag = J_diag.at[1:].add(J_22)
        J_lower_diag = J_21
        return J_diag, J_lower_diag


    # Find the mean and precision of the Laplace approximation
    mu = _compute_laplace_mean(initial_states)

    # The precision is given by the negative hessian at the mode
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(mu)

    # Compute the linear potential by multiplying a block tridiagonal matrix with a vector
    # We represent the block tridiag matrix with the (T, D, D) array of diagonal blocks
    # and the (T-1, D, D) array of lower diagonal blocks. The vector is represented
    # as a (T, D) array.
    f = vmap(jnp.matmul)
    h = f(J_diag, mu) # (T, D)
    h = h.at[1:].add(f(J_lower_diag, mu[:-1]))
    h = h.at[:-1].add(f(jnp.swapaxes(J_lower_diag, -1, -2), mu[1:]))

    log_normalizer, Ex, ExxT, ExxnT = block_tridiag_mvn_expectations(J_diag, J_lower_diag, h)

    # Returns log_normalizer, Ex, ExxT, ExxnT, and posterior params for sampling
    return log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h


def fit_laplace_em(slds, key, emissions, initial_zs, initial_xs,
                    num_iters=100, n_discrete_samples=1):
    """
    Estimate the parameters of the SLDS and an approximate posterior distr.
    over latent states using Laplace EM. Specifically, the approximate
    posterior factors over discrete and continuous latent states. The
    discrete state posterior is a discrete chain graph, and the continuous
    posterior is a linear Gaussian chain. We estimate the continuous posterior
    using a Laplace approximation, which is appropriate when the likelihood
    is log concave in the continuous states.
    """
    K = slds.num_states
    D = slds.latent_dim
    N = slds.emission_dim
    ys = emissions
    B, T, _ = ys.shape

    def _update_discrete_states(slds, key, J_diag, J_lower_diag, h):
        """
        Update the discrete states to the coordinate-wise maximum using the
        Viterbi algorithm.
        """
        # sample xs from q(x)
        key, *skeys = jr.split(key, n_discrete_samples+1)
        vmap_block_tridiag_mvn_sample = vmap(block_tridiag_mvn_sample, in_axes=(0, None, None, None))
        x_samples = vmap_block_tridiag_mvn_sample(jnp.array(skeys), J_diag, J_lower_diag, h)

        pi0 = jnp.mean(jnp.array(
            [slds.pi0
                for x in x_samples]), axis=0)

        # TODO: eventually, transition matrix will depend on x
        P = jnp.mean(jnp.array(
            [slds.transition_matrix
                for x in x_samples]), axis=0)

        def _dynamics_likelihood(xs):
            f0 = lambda z: slds.init_continuous_state_distn(z).log_prob(xs[0])
            f = lambda z: vmap(lambda zn, x, xn: slds.dynamics_distn(zn, x).log_prob(xn), in_axes=(None, 0, 0))(z, xs[:-1], xs[1:]) # T-1
            return jnp.vstack([
                vmap(f0)(jnp.arange(K)),
                vmap(f)(jnp.arange(K)).T   # (T-1, K)
            ])

        log_likes = jnp.mean(vmap(_dynamics_likelihood)(x_samples), axis=0)

        return hmm.inference.hmm_smoother(pi0, P, log_likes), x_samples[0]

    def _update_continuous_states(slds, ys, zs, xs):

        # Define log prob functions that close over zs
        log_prob = lambda xs, ys: slds.log_prob(ys, zs, xs)

        # TODO : change these to slds object distributions
        # TODO : marginalize over q(z)
        initial_distribution = lambda x0: slds.init_continuous_state_distn(zs[0]).log_prob(x0)
        dynamics_distribution = lambda t, xt, xtp1: slds.dynamics_distn(zs[t+1], xt).log_prob(xtp1)
        emission_distribution = lambda t, xt, yt: slds.emission_distn(xt).log_prob(yt)
        log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = \
            laplace_approximation(log_prob,
                                initial_distribution,
                                dynamics_distribution,
                                emission_distribution,
                                jnp.zeros_like(xs),
                                ys,
                                method="L-BFGS",
                                num_iters=100)

        return Ex, ExxT, ExxnT, J_diag, J_lower_diag, h

    def _update_params(slds, ys, zs, xs, lr=1e-3, num_iters=10):

        def _objective(slds):
            return -1.0 * jnp.sum(vmap(slds.log_prob)(ys, zs, xs)) / ys.size

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(slds)

        def _step(carry, args):
            slds, opt_state = carry
            grads = jax.grad(_objective)(slds)
            updates, opt_state = optimizer.update(grads, opt_state)
            slds = optax.apply_updates(slds, updates)
            return (slds, opt_state), None

        (slds, _), _ = lax.scan(_step, (slds, opt_state), None, length=num_iters)

        return slds

    def _step(carry, step_size):
        zs, xs, slds, key = carry
        # lp = vmap(slds.log_prob)(ys, zs, xs).sum()
        Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = lax.map(lambda args : _update_continuous_states(slds, *args), (ys, zs, xs))
        xs = Ex # redefine xs as mean
        key, skey = jr.split(key)
        post, x_samples = vmap(partial(_update_discrete_states, slds))(jr.split(skey, B), J_diag, J_lower_diag, h)
        zs = jnp.argmax(post.smoothed_probs, axis=-1)
        slds = _update_params(slds, ys, zs, xs, lr=1e-3, num_iters=10)
        lp = vmap(slds.log_prob)(ys, zs, xs).sum()
        return (zs, xs, slds, key), lp

    initial_carry = (initial_zs, initial_xs, slds, key)
    step_sizes = jnp.ones((num_iters))
    (zs, xs, slds, key), lps = lax.scan(_step, initial_carry, step_sizes)

    # carry = initial_carry
    # for i in range(num_iters):
    #     carry, out = _step(carry, step_sizes[i])

    return slds, lps, zs, xs, key