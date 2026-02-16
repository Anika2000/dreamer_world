import torch


def imagination_rollout(rssm, actor, start_h, start_z, horizon=15):
    """
    Rollout imagined trajectories in latent space using current actor.
    """
    h, z = start_h, start_z
    imagined_h, imagined_z, actions = [], [], []

    for t in range(horizon):
        a = actor(h, z)                 # action from policy
        h, z, _, _, _ = rssm(h, z, a, embed=None, use_relaxed=True)  # imagined next latent
        imagined_h.append(h)
        imagined_z.append(z)
        actions.append(a)

    # Stack tensors along time dimension: (T, B, dim)
    imagined_h = torch.stack(imagined_h, dim=0)
    imagined_z = torch.stack(imagined_z, dim=0)
    actions = torch.stack(actions, dim=0)

    return imagined_h, imagined_z, actions

def compute_actor_critic_loss(value_head, imagined_h, imagined_z, gamma=0.99, lam=0.95):
    """
    Compute actor + critic losses along imagined trajectories.
    """
    horizon, batch_size, _ = imagined_h.shape
    device = imagined_h.device

    # Compute predicted values at each imagined step
    values = torch.stack([value_head(h, z) for h, z in zip(imagined_h, imagined_z)], dim=0)
    values = values.squeeze(-1)  # (T, B)

    # Compute lambda-returns
    returns = torch.zeros_like(values)
    G = torch.zeros(batch_size, device=device)
    for t in reversed(range(horizon)):
        r = torch.zeros(batch_size, device=device)  # reward head not used here, could add if desired
        G = r + gamma * ((1 - lam) * values[t] + lam * G)
        returns[t] = G

    # Critic loss = MSE of value predictions vs returns
    critic_loss = ((values - returns)**2).mean()

    # Actor loss = maximize imagined returns (gradient ascent)
    actor_loss = -returns.mean()

    return actor_loss, critic_loss
