import torch
from torch.distributions import Distribution as torchDist
from typing import Callable, Dict, Optional, Tuple
from tqdm import trange

def mala(
    start: torch.FloatTensor,
    target: torchDist,
    proposal,
    n_samples: int,
    burn_in: int,
    project: Callable = lambda x: x,
    *,
    step_size: float,
    verbose: bool = False,
    meta: Optional[Dict] = None,
) -> Tuple[torch.FloatTensor, Dict]:
    """
    Metropolis-Adjusted Langevin Algorithm with Normal proposal

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        step_size - step size for drift term
        verbose - whether show iterations' bar

    Returns:
        sequence of slices per each iteration, meta
    """
    if n_samples + burn_in <= 0:
        raise ValueError("Number of steps might be positive")

    chains = []
    point = start.clone()
    point.requires_grad_()
    point.grad = None

    device = point.device
    proposal = torch.distributions.MultivariateNormal(
        torch.zeros(point.shape[-1], device=device),
        torch.eye(point.shape[-1], device=device),
    )

    meta = meta or dict()
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])

    meta["logp"] = logp_x = target.log_prob(
        point
    )
    if "grad" not in meta:
        grad_x = torch.autograd.grad(logp_x.sum(), point)[0].detach()
        meta["grad"] = grad_x
    else:
        grad_x = meta["grad"]

    pbar = trange if verbose else range
    for step_id in pbar(n_samples + burn_in):
        noise = proposal.sample(point.shape[:-1])
        proposal_point = point + step_size * grad_x + noise * (2 * step_size) ** 0.5
        proposal_point = project(proposal_point)
        proposal_point = proposal_point.detach().requires_grad_()

        logp_y = target.log_prob(proposal_point)
        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
            create_graph=False,
            retain_graph=False,
        )[
            0
        ]  # .detach()

        log_qyx = proposal.log_prob(noise)
        log_qxy = proposal.log_prob(
            (point - proposal_point - step_size * grad_y) / (2 * step_size) ** 0.5
        )

        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob
        mask = mask.detach()

        with torch.no_grad():
                point[mask, :] = proposal_point[mask, :]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]

        meta["mh_accept"].append(mask.float().mean().item())
        meta["step_size"].append(step_size)

        point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point.cpu().clone())
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["grad"] = grad_x
    meta["mask"] = mask.detach().cpu()

    return chains, meta