import torch
import numpy as np
from torch.distributions import Distribution as torchDist
from torch.distributions import Categorical
from typing import Callable, Dict, Optional, Tuple, Union
from torch.nn import functional as F
from tqdm import trange
from mala import mala
from isir import isir_step

def ex2mcmc(
    start: torch.FloatTensor,
    target: torchDist,
    proposal: torchDist,
    n_samples: int,
    burn_in: int,
    step_size: float,
    n_particles: int,
    project: Callable = lambda x: x,
    n_mala_steps: int = 1,
    verbose: bool = False,
    meta: Optional[Dict] = None,
):
    """
    Ex2MCMC

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        proposal - proposal distribution
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_size - step size for drift term
        n_particles - number of particles including one from previous step
        n_mala_steps - number of MALA steps after each SIR step
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim],
          acceptance rates for each iteration
    """

    chains = []
    point = start.clone()
    point.requires_grad_(True)
    point.grad = None

    meta = meta or dict()
    meta["sir_accept"] = meta.get("sir_accept", [])
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])

    pbar = trange if verbose else range

    meta["logp"] = meta.get("logp", target.log_prob(point))

    for step_id in pbar(n_samples + burn_in):
        logp_x = meta["logp"]
        logq_x = proposal.log_prob(point)

        point, proposals, log_ps, log_qs, indices = isir_step(
            point,
            target,
            proposal,
            n_particles=n_particles,
            logq_x=logq_x,
        )
        logp_x = log_ps[np.arange(point.shape[0]), indices]
        meta["sir_accept"].append((indices != 0).float().mean().item())
        point = point.detach().requires_grad_()
        meta["logp"] = logp_x
        meta["mask"] = (
            F.one_hot(indices, num_classes=n_particles).to(bool).detach().cpu()
        )

        points, meta = mala(
            point,
            target,
            proposal,
            n_mala_steps,
            n_mala_steps - 1,
            project,
            step_size=step_size,
            meta=meta,
        )
        step_size = meta["step_size"][-1]
        point = points[-1].to(point.device)
        point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point.cpu().clone())

    chains = torch.stack(chains, 0)
    return chains, meta
