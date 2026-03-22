import torch
import torch.nn as nn
import torch.distributions as D

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
        image_loss: reconstruction loss
        reward_loss: reward prediction loss
        kl_loss: KL divergence loss
        discount loss
    """
    # Assuming out["reconstructed"] is the mean of a Gaussian distribution
    pixel_dist = D.Normal(out["reconstructed"], 1.0)  # fixed std dev = 1.0 or learned
    # Reconstruction loss (mean squared error)
    image_loss = -pixel_dist.log_prob(obs).mean()
    
    # Reward prediction loss (MSE)
    reward_loss = ((out["reward"] - reward) ** 2).mean()
    
    #discount loss (Bernoulli likelihood)
    discount_loss = nn.functional.binary_cross_entropy_with_logits(
        out["discount"], discount
    )
    # print("recon min/max:", out["reconstructed"].min().item(), out["reconstructed"].max().item())
    # print("obs min/max:", obs.min().item(), obs.max().item())

    #########################
    # KL Balancing: # KL divergence between posterior and prior
    #########################
    post_logits = out["post_dist"].logits
    prior_logits = out["prior_dist"].logits

    # Create distributions
    post_dist = D.Categorical(logits=post_logits)
    prior_dist = D.Categorical(logits=prior_logits)

    # -----------------------------
    # KL BALANCING (correct)
    # -----------------------------

    # IMPORTANT: no .mean() yet — keep elementwise KL
    kl_prior = D.kl_divergence(
        D.Categorical(logits=post_logits.detach()), prior_dist
    )

    kl_post = D.kl_divergence(
        post_dist, D.Categorical(logits=prior_logits.detach())
    )

    # Balanced KL (Dreamer-style)
    kl_balanced = alpha * kl_prior + (1 - alpha) * kl_post

    # -----------------------------
    # FREE NATS (correct placement)
    # -----------------------------
    free_nats = 0.01

    kl_balanced = torch.maximum(
        kl_balanced,
        torch.tensor(free_nats, device=kl_balanced.device)
    )

    # NOW reduce to scalar
    kl_loss = kl_balanced.mean()

    # Total loss
    total_loss = image_loss + reward_loss + discount_loss + beta * kl_loss
    return total_loss, image_loss, reward_loss, kl_loss, discount_loss


