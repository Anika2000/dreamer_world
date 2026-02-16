import torch

def world_model_loss(out, obs, reward, free_nats=3.0):
    """
    Computes the losses for the WorldModel.
    
    Args:
        out: dict of WorldModel outputs (reconstructed, reward, post_dist, prior_dist)
        obs: ground truth observations (B, C, H, W)
        reward: ground truth rewards (B, 1)
        free_nats: minimum KL divergence allowed
    
    Returns:
        total_loss: scalar tensor combining all losses
        obs_loss: reconstruction loss
        reward_loss: reward prediction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (mean squared error)
    obs_loss = ((out["reconstructed"] - obs) ** 2).mean()
    
    # Reward prediction loss (MSE)
    reward_loss = ((out["reward"] - reward) ** 2).mean()
    
    # KL divergence between posterior and prior
    kl = torch.distributions.kl_divergence(out["post_dist"], out["prior_dist"])
    kl_loss = torch.clamp(kl.mean(), min=free_nats)
    
    # Total loss
    total_loss = obs_loss + reward_loss + kl_loss
    return total_loss, obs_loss, reward_loss, kl_loss
