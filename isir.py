import torch
import numpy as np
from torch.distributions import Distribution as torchDist
from torch.distributions import Categorical
from typing import Tuple

def isir_step(
    start: torch.FloatTensor,
    target,
    proposal: torchDist,
    *,
    n_particles: int,
    logq_x=None,
) -> Tuple:
    point = start.clone()
    logq_x = proposal.log_prob(point) if logq_x is None else logq_x

    particles = proposal.sample((point.shape[0], n_particles - 1))
    log_qs = torch.cat([logq_x[:, None], proposal.log_prob(particles)], 1)
    particles = torch.cat([point[:, None, :], particles], 1)
    log_ps = target.log_prob(particles)

    log_weights = log_ps - log_qs
    indices = Categorical(logits=log_weights).sample()

    x = particles[np.arange(point.shape[0]), indices]

    return x, particles, log_ps, log_qs, indices
