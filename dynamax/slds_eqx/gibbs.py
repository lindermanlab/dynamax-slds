import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as tree
import operator
import optax 

from jax import grad, lax, vmap
from jaxtyping import Array, Float
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm
from dynamax import linear_gaussian_ssm as lgssm
from .models import SLDS

def fit_gibbs(slds : SLDS, 
              key : jr.PRNGKey, 
              emissions : Float[Array["num_timesteps emission_dim"]], #array of floats of dim num_timesteps x emission_dim
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
    ys = emissions # num_timesteps x emission_dim

    #theta denotes parameters of the model
    #z denotes discrete latent states
    #x denotes continuous latent states
    #y denotes emissions

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
        lls = vmap(f, jnp.arange(K)).T                                                              # (T-1,K)

        # Stack the initial log prob and subsequent log probs into one array
        lls = jnp.vstack([ll0, lls])
    
        return hmm.inference.hmm_posterior_sample(key1, pi0, P, lls)

    def _update_continuous_states(slds, key2, ys, zs):
        # TODO: sample from the p(x | z, y) by using lgssm_posterior_sample 
        # and giving the function time-varying parameters A_t = A_{z_t}
        """
        Update the continuous states by drawing a sample from p(x | z, y)
        """
        T = ys.shape[0] #number of timesteps

        # Initialize time-varying parameters
        As = slds.dynamics_matrices
        bs = slds.dynamics_biases
        Qs = slds.dynamics_covs
        C = slds.emission_matrix
        d = slds.emission_bias
        R = slds.emission_cov

        # Compute parameters for each time step using the discrete states
        A_t = vmap(lambda z: As[z])(zs)
        b_t = vmap(lambda z: bs[z])(zs)
        Q_t = vmap(lambda z: Qs[z])(zs)

        # Create ParamsLGSSM object to pass into lgssm_posterior_sample
        params = lgssm.inference.ParamsLGSSM(
            initial= lgssm.inference.ParamsLGSSMInitial(
                mean=jnp.zeros(D),
                cov=jnp.eye(D) #identity matrix - assumes initial latent dimensions are uncorrelated
            ),
            dynamics=lgssm.inference.ParamsLGSSMDynamics(
                #ntime x state_dim x state_dim
                weights=A_t,
                bias=b_t,
                input_weights=None,
                cov=Q_t
            ),
            emissions=lgssm.inference.ParamsLGSSMEmissions(
                #broadcasting C, d, R to have shape (T, N, D), (T, N), (T, N, N)
                weights=jnp.repeat(C[None, :, :], T, axis=0),
                bias=jnp.repeat(d[None, :], T, axis=0),
                input_weights=None,
                cov=jnp.repeat(R[None, :, :], T, axis=0)
            ),
        )

        # Sample from the posterior distribution
        xs = lgssm.inference.lgssm_posterior_sample(key2, params, ys)

        return xs
    
    def _update_params(slds, ys, zs, xs, lr=1e-3, reg=1.0, num_iters=10):
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
        
        # Minimize the loss with optax
        # TODO: replace for loop with a scan
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(slds)
        for _ in range(num_iters):
            grads = grad(loss)(slds)
            updates, opt_state = optimizer.update(grads, opt_state)
            slds = optax.apply_updates(slds, updates)
        return slds

    def _step(carry, step_size): #not using step_size here (num_iters is used instead)
        # TODO
        # 1. call _update_discrete_states
        # 2. call _update_continuous_states
        # 3. compute the log joint probability (using slds.log_prob)
        # 4. return new_carry and output lp

        # Unpack Carry
        zs, xs, slds, key = carry

        # Update Key to generate new random samples
        key, subkey1, subkey2 = jr.split(key, 3)

        # Update Discrete States p(z₁:ₜ | x₁:ₜ, θ)
        zs = _update_discrete_states(slds, subkey1, xs)

        # Update Continuous States p(x₁:ₜ | z₁:ₜ, y₁:ₜ, θ)
        xs = _update_continuous_states(slds, subkey2, ys, zs)

        # Compute Log Joint Probability log p(y₁:ₜ, z₁:ₜ, x₁:ₜ | θ)
        lp = slds.log_prob(ys, zs, xs)

        # Update Parameters
        slds = _update_params(slds, ys, zs, xs)

        # Return New Carry and Output Log Probability
        new_carry = (zs, xs, slds, key)

        return new_carry, lp

    # TODO: initialize carry and call scan
    initial_carry = (initial_zs, initial_xs, slds, key)
    final_carry, lps = lax.scan(_step, initial_carry, jnp.arange(num_iters)) #step_size is num_iters

    # Unpack Final Carry
    zs, xs, slds, key = final_carry

    return slds, lps, zs, xs
