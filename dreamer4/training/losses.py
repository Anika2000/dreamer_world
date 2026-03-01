import torch
import torch.nn as nn

def world_model_loss(out, obs, reward, discount, beta=1.0, alpha=0.8):
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
    
    #discount loss (Bernoulli likelihood)
    discount_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["discount"], discount
    )

    #########################
    # KL Balancing: # KL divergence between posterior and prior
    #########################
    post_probs = out["post_dist"].probs
    prior_probs = out["prior_dist"].probs
    # KL where gradient flows only into prior
    kl_prior = (post_probs.detach() * (post_probs.detach().log() - prior_probs.log())).sum(-1).mean()


    # KL where gradient flows only into posterior
    kl_post = (post_probs * (post_probs.log() - prior_probs.detach().log())).sum(-1).mean()

    # Balanced KL (Section 2.1)
    kl_balanced = alpha * kl_prior + (1 - alpha) * kl_post

    #Free nats are a minimum threshold on KL:
    #Without this, your latent can collapse (posterior becomes equal to prior, ignoring the observation).
    free_nats = 3.0  # usually 3 nats
    kl_loss = torch.clamp(kl_balanced, min=free_nats)
    # Total loss
    total_loss = obs_loss + reward_loss + discount_loss + beta * kl_loss
    return total_loss


