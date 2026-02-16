import torch
import torch.nn as nn

def world_model_loss(out, obs, reward, discount=None, beta=1.0, free_nats=3.0):
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
        discount loss
    """
    # Reconstruction loss (mean squared error)
    obs_loss = ((out["reconstructed"] - obs) ** 2).mean()
    
    # Reward prediction loss (MSE)
    reward_loss = ((out["reward"] - reward) ** 2).mean()
    
    # KL divergence between posterior and prior
    post_probs = out["post_dist"].probs
    prior_probs = out["prior_dist"].probs
    kl = (post_probs * (post_probs.log() - prior_probs.log())).sum(-1).mean()
    kl = torch.clamp(kl, min=free_nats)
    kl_loss = beta * kl

        # Discount loss 
    if discount is not None:
        pred_discount = out["discount"]
        discount_loss = nn.functional.mse_loss(pred_discount, discount)
    else:
        discount_loss = 0.0
    
    # Total loss
    total_loss = obs_loss + reward_loss + kl_loss + discount_loss
    return total_loss, obs_loss, reward_loss, kl_loss, discount_loss


