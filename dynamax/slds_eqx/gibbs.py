import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as tree
import operator
import optax 

from jax import grad, lax, vmap
from jaxtyping import Array, Float
from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
import jax.random as jr
from typing import Callable
tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm
from dynamax import linear_gaussian_ssm as lgssm
from .models import SLDS

def fit_gibbs(slds : SLDS, 
              key : jr.PRNGKey, 
              emissions : jnp.ndarray,
              initial_zs : jnp.ndarray,
              initial_xs :jnp.ndarray,
              num_iters : int = 100,
              lr : float = 1e-3,
              reg_schedule : Callable[[int], float] = lambda t: 1.0,
              param_update_iters : int = 10
              ):
    """
    Run a Gibbs sampler to draw (approximate) samples from the posterior distribution over
    discrete and continuous latent states of an SLDS.
    """
    K = slds.num_states 
    D = slds.latent_dim
    N = slds.emission_dim
    ys = emissions

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(slds)

    def _update_discrete_states(slds, key1, xs):
        """
        Update the discrete states by drawing a sample from p(z | x, theta)

        Note: the discrete states (z) are conditionally independent of the emissions (y)
        given the continuous latent states (x).
        """
        pi0 = slds.pi0
        P = slds.transition_matrix

        # log p(x_1 | z_1=k) for all k=1,...,K
        ll0 = vmap(lambda z0: slds.init_continuous_state_distn(z0).log_prob(xs[0]))(jnp.arange(K))  # (K,)
        
        # log p(x_t | x_{t-1}, z_t=k) for all t=2,...,T and all k=1,...,K
        f = lambda z: vmap(lambda x, xn: slds.dynamics_distn(z, x).log_prob(xn))(xs[:-1], xs[1:])   # [K] -> (T-1,)
        lls = vmap(f)(jnp.arange(K)).T                                                            # (T-1,K)

        # Stack the initial log prob and subsequent log probs into one array
        lls = jnp.vstack([ll0, lls])

        return hmm.inference.hmm_posterior_sample(key1, pi0, P, lls)[1]

    def _update_continuous_states(slds, key2, ys, zs):
        """
        Update the continuous states by drawing a sample from p(x | z, y)
        """

        # Initialize time-varying parameters
        As = slds.dynamics_matrices
        bs = slds.dynamics_biases
        Qs = slds.dynamics_covs
        C = slds.emission_matrix
        d = slds.emission_bias
        R = slds.emission_cov
        initial_mean = jnp.zeros(D)
        initial_cov = jnp.eye(D)

        # Compute parameters for each time step using the discrete states
        A_t = As[zs]
        b_t = bs[zs]
        Q_t = Qs[zs]

        params = lgssm.inference.make_lgssm_params(
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            dynamics_weights=A_t,
            dynamics_cov=Q_t,
            emissions_weights=C,
            emissions_cov=R,
            dynamics_bias=b_t,
            dynamics_input_weights=None,
            emissions_bias=d,
            emissions_input_weights=None,
        )

        # Sample from the posterior distribution
        xs = lgssm.inference.lgssm_posterior_sample(key2, params, ys)

        return xs
    
    def _update_params(slds, ys, zs, xs, opt_state, reg=1.0, num_iters=10):
        r"""
        Goal: maximize the expected log probability as a function of parameters \theta:
            L(\theta) = E_{p(z, x | y, \theta')}[log p(y, z, x; \theta)]

        We can't compute the posterior exactly, so instead we'll approximate it with
        Monte Carlo, using Gibbs to generate samples of z and x and then 
        maximize the approximate objective,
            
            \tilde{L}(\theta) = \frac{1}{S} \sum_s log p(y, z_s, x_s; \theta)

        where the latent states are (approximately) sampled from the posterior

            z_s, x_s \sim p(z, x | y, \theta')

        In practice, we are setting S = 1 (i.e., using a single sample of the latents).

        Technically, to guarantee convergence we need to add an additional constraint.
        Namely, we can't let the parameters change too much from one iteration to the
        next, so we include a regularizer

            R(\theta) = \frac{\alpha}{2} \|\theta - \theta'\|_2^2

        The final objective combines these two terms.
        """

        T = ys.shape[0]
        def loss(curr_slds):
            L = -1 * curr_slds.log_prob(ys, zs, xs) / T
            L += 0.5 * reg * tree.reduce(
                operator.add,
                tree.map(lambda x, y: jnp.sum((x - y)**2), curr_slds, slds),
                0.0)
            return L

        # Define a single step of the optimization
        def step(carry, _):
            curr_slds, opt_state = carry
            grads = grad(loss)(curr_slds)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_slds = optax.apply_updates(curr_slds, updates)
            return (new_slds, new_opt_state), None

        # Run the optimization using lax.scan
        (final_slds, final_opt_state), _ = lax.scan(step, (slds, opt_state), None, length=num_iters)

        return final_slds, final_opt_state

    def _step(carry, t):
        # Unpack Carry
        zs, xs, slds, opt_state, key = carry

        # Update Key to generate new random samples
        key, subkey1, subkey2 = jr.split(key, 3)

        # Update Discrete States p(z₁:ₜ | x₁:ₜ, θ)
        zs = _update_discrete_states(slds, subkey1, xs)

        # Update Continuous States p(x₁:ₜ | z₁:ₜ, y₁:ₜ, θ)
        xs = _update_continuous_states(slds, subkey2, ys, zs)

        # Compute Log Joint Probability log p(y₁:ₜ, z₁:ₜ, x₁:ₜ | θ)
        lp = slds.log_prob(ys, zs, xs)

        # Compute regularization strength for this iteration
        reg = reg_schedule(t)

        # Update Parameters
        slds, opt_state = _update_params(slds, ys, zs, xs, opt_state, reg, param_update_iters)

        # Return New Carry and Output Log Probability
        new_carry = (zs, xs, slds, opt_state, key)

        return new_carry, lp

    initial_carry = (initial_zs, initial_xs, slds, opt_state, key)
    final_carry, lps = lax.scan(_step, initial_carry, jnp.arange(num_iters))

    # Unpack Final Carry
    zs, xs, slds, _, _ = final_carry

    return slds, lps, zs, xs
