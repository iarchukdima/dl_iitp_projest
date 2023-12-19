import torch
from ex2mcmc import ex2mcmc
from torch.distributions import MultivariateNormal as MNormal

class Ex2mcmc():

    def __init__(self, dim, proposal, n_samples, burn_in, step_size, n_particles, n_mala_steps, verbose=True) -> None:

        self._start        = torch.randn((1, dim))
        self._dim          = dim
        self._device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._proposal     = proposal
        self._target       = MNormal(torch.zeros(dim, device=self._device), 1 ** 2 * torch.eye(dim, device=self._device))
        self._n_samples    = n_samples
        self._burn_in      = burn_in
        self._step_size    = step_size
        self._n_particles  = n_particles
        self._n_mala_steps = n_mala_steps
        self._verbose      = verbose
    
    def dim(self):
        return self._dim
    
    def run(self):
        chains, meta = ex2mcmc(
                       start=self._start, 
                       target=self._target, 
                       proposal=self._proposal, 
                       n_samples=self._n_samples, 
                       burn_in=self._burn_in, 
                       step_size=self._step_size, 
                       n_particles=self._n_particles, 
                       n_mala_steps=self._n_mala_steps,
                       verbose=self._verbose
                       )
        chains = chains.detach().cpu()
        return chains
