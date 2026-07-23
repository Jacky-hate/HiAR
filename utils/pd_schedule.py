"""Schedule alignment helpers shared by ODE generation and training tests."""

import torch


def shifted_student_sigmas(steps, shift, train_timesteps=1000):
    """Map nominal student timesteps to the actual flow-matching sigmas."""
    sigmas = []
    for timestep in steps:
        raw_sigma = float(timestep) / train_timesteps
        sigma = shift * raw_sigma / (1 + (shift - 1) * raw_sigma)
        sigmas.append(sigma)
    if not sigmas or sigmas[-1] != 0.0:
        sigmas.append(0.0)
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError(f"student sigmas must be strictly decreasing: {sigmas}")
    return sigmas


def interpolate_trajectory(trajectory, trajectory_sigmas, target_sigmas):
    """Linearly sample a piecewise-Euler ODE path at exact target sigmas."""
    outputs = []
    sigma_values = trajectory_sigmas.detach().double().cpu()
    for target in target_sigmas:
        target = float(target)
        distances = (sigma_values - target).abs()
        nearest = int(distances.argmin().item())
        if distances[nearest].item() <= 1.0e-8:
            outputs.append(trajectory[:, nearest])
            continue
        lower_candidates = torch.nonzero(
            sigma_values < target, as_tuple=False
        ).flatten()
        if lower_candidates.numel() == 0 or lower_candidates[0].item() == 0:
            raise ValueError(
                f"target sigma {target} is outside teacher trajectory range"
            )
        lower_index = int(lower_candidates[0].item())
        upper_index = lower_index - 1
        upper_sigma = float(sigma_values[upper_index])
        lower_sigma = float(sigma_values[lower_index])
        weight = (upper_sigma - target) / (upper_sigma - lower_sigma)
        outputs.append(
            trajectory[:, upper_index]
            + weight
            * (trajectory[:, lower_index] - trajectory[:, upper_index])
        )
    return torch.stack(outputs, dim=1)
