import jax.numpy as jnp
import jax.random as jr

from jax import vmap, lax
from jaxtyping import Array, Float
from tensorflow_probability.substrates import jax as tfp

import equinox as eqx

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag


class SLDS(eqx.Module):
    """
    A switching linear dynamical system

    Note: Assume that C, d, R are shared by all states
    Note: Assume that initial distribution is uniform over discrete state
          and the standard normal over the continuous latent state
    """
    num_states : int = eqx.static_field()
    latent_dim : int = eqx.static_field()
    emission_dim : int = eqx.static_field()

    transition_logits : Float[Array, "num_states num_states-1"]
    dynamics_matrices : Float[Array, "num_states latent_dim latent_dim"] # For each state this is latent_dim x latent_dim
    dynamics_biases : Float[Array, "num_states latent_dim"]
    dynamics_diag_logvars : Float[Array, "num_states latent_dim"]
    emission_matrix : Float[Array, "emission_dim latent_dim"]
    emission_bias : Float[Array, "emission_dim"]
    emission_diag_logvars : Float[Array, "emission_dim"]

    def __init__(self,
                 num_states : int,
                 latent_dim : int,
                 emission_dim : int,
                 log_P, As, bs, log_Qs, C, d, log_R):
        # TODO: move parameter initialization to here
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim

        self.transition_logits = log_P
        self.dynamics_matrices = As
        self.dynamics_biases = bs
        self.dynamics_diag_logvars = log_Qs
        self.emission_matrix = C
        self.emission_bias = d
        self.emission_diag_logvars = log_R

    @property
    def pi0(self):
        return jnp.ones((self.num_states,)) / self.num_states

    def init_continuous_state_distn(self, z0):
        return MVN(jnp.zeros(self.latent_dim), jnp.eye(self.latent_dim))

    def init_discrete_state_distn(self):
        return tfd.Categorical(probs=self.pi0)

    @property
    def transition_matrix(self):
        return tfb.SoftmaxCentered().forward(self.transition_logits)

    # @transition_matrix.setter
    # def transition_matrix(self, P):
    #     self.transition_logits = tfb.SoftmaxCentered().inverse(P)

    def transition_distn(self, z):
        """
        Currently z can be a scalar or a sequence
        """
        P = self.transition_matrix
        return tfd.Categorical(probs=P[z])

    @property
    def dynamics_covs(self):
        return vmap(jnp.diag)(tfb.Softplus(low=1e-5).forward(self.dynamics_diag_logvars))

    # @dynamics_covs.setter
    # def dynamics_covs(self, Q):
    #     self.dynamics_diag_logvars = tfb.Softplus(low=1e-5).inverse(vmap(jnp.diag)(Q))

    def dynamics_distn(self, z, x):
        As = self.dynamics_matrices
        bs = self.dynamics_biases
        Qs = self.dynamics_covs
        return MVN(As[z] @ x + bs[z], Qs[z])

    @property
    def emission_cov(self):
        return jnp.diag(tfb.Softplus(low=1e-5).forward(self.emission_diag_logvars))

    # @emission_cov.setter
    # def emission_cov(self, R):
    #     self.emission_diag_logvars = tfb.Softplus(low=1e-5).inverse(jnp.diag(R))

    def emission_distn(self, x):
        C = self.emission_matrix
        d = self.emission_bias
        R = self.emission_cov
        return MVN(C @ x + d, R)

    def log_prob(self, ys, zs, xs):
        # prior
        lp = tfd.Dirichlet(concentration=1.1*jnp.ones((self.num_states,))).log_prob(self.transition_matrix).sum()

        # TODO : add initial time point log prob
        # lp = self.init_discrete_state_distn().log_prob(zs[0])
        # lp += self.init_continuous_state_distn(zs[0]).log_prob(xs[0])
        # log p(z_t | z_{t-1})
        lp += vmap(lambda z, zn : self.transition_distn(z).log_prob(zn))(zs[:-1], zs[1:]).sum()
        # log p(x_t | A_z x_{t-1} + b_z, Q_z)
        lp += vmap(lambda zn, x, xn : self.dynamics_distn(zn, x).log_prob(xn))(zs[1:], xs[:-1], xs[1:]).sum()
        # log p(y_t | C x_t + d, R)
        lp += vmap(lambda x, y : self.emission_distn(x).log_prob(y))(xs, ys).sum()
        return lp

    def sample(self,
               key : jr.PRNGKey,
               num_timesteps : int):
        """

        """
        # Sample the first time steps
        k1, k2, k3, key = jr.split(key, 4)
        z0 = self.init_discrete_state_distn().sample(seed=k1)
        x0 = self.init_continuous_state_distn(z0).sample(seed=k2)
        y0 = self.emission_distn(x0).sample(seed=k3)

        def _step(carry, key):
            zp, xp, yp = carry
            k1, k2, k3, key = jr.split(key, 4)

            z = self.transition_distn(zp).sample(seed=k1)
            x = self.dynamics_distn(z, xp).sample(seed=k2)
            y = self.emission_distn(x).sample(seed=k3)
            return (z, x, y), (zp, xp, yp)

        _, (zs, xs, ys) = lax.scan(_step, (z0, x0, y0), jr.split(key, num_timesteps))
        return zs, xs, ys

