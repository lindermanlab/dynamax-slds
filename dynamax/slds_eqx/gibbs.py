import jax
import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from jaxtyping import Array, Float
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm

from .models import SLDS

def fit_gibbs(slds : SLDS, 
              key : jr.PRNGKey, 
              emissions : Float[Array["num_timesteps emission_dim"]], 
              initial_zs : Float[Array["num_timesteps"]], 
              initial_xs : Float[Array["num_timesteps latent_dim"]], 
              num_iters : int = 100
              ):
    """
    Run a Gibbs sampler to draw (approximate) samples from the posterior distribution over
    discrete and continuous latent states of an SLDS.
    """
    K = slds.num_states
    D = slds.latent_dim
    N = slds.emission_dim
    ys = emissions

    def _update_discrete_states(slds, key, xs):
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
        lls = vmap(f, jnp.arange(K)).T                                                              # (T-1,K)

        # Stack the initial log prob and subsequent log probs into one array
        lls = jnp.vstack([ll0, lls])
    
        return hmm.inference.hmm_posterior_sample(key, pi0, P, lls)

    def _update_continuous_states(slds, ys, zs):
        # TODO: sample from the p(x | z, y) by using lgssm_posterior_sample 
        # and giving the function time-varying parameters A_t = A_{z_t}

        return xs
    
    def _update_params(slds, ys, zs, xs, lr=1e-3, num_iters=10):
        return slds

    def _step(carry, step_size):
        # TODO
        # 1. call _update_discrete_states
        # 2. call _update_continuous_states
        # 3. compute the log joint probability (using slds.log_prob)
        # 4. return new_carry and output lp
        raise NotImplementedError

    # TODO: initialize carry and call scan
    # initial_carry = (initial_zs, initial_xs, slds, key)
    # final_carry, lps = lax.scan(_step, initial_carry, step_sizes)

    
    return slds, lps, zs, xs
