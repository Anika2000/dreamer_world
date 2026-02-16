import torch.nn as nn 
from dreamer4.models.encoder import Encoder
from dreamer4.models.rssm import RSSM
from dreamer4.models.decoder import Decoder
from dreamer4.models.heads import RewardHead, ValueHead, DiscountHead
class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.rssm = RSSM(
            action_dim=config["env"]["action_dim"],
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
            categories=config["model"]["categories"]
        )
        self.decoder = Decoder(config)
        self.reward_head = RewardHead(
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
            categories=config["model"]["categories"]
        )
        self.value_head = ValueHead(
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
            categories=config["model"]["categories"]
        )
        self.discount_head = DiscountHead(
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
            categories=config["model"]["categories"]
        )

    def forward(self, obs, prev_a, prev_h, prev_z, use_relaxed=False, temperature=0.67):
        embed = self.encoder(obs)
        h, z, z_prior, post_dist, prior_dist = self.rssm(prev_h, prev_z, prev_a, embed,
                                                         use_relaxed=use_relaxed, temperature=temperature)
        recon = self.decoder(h, z)
        pred_reward = self.reward_head(h, z)
        pred_value = self.value_head(h, z)
        pred_discount = self.discount_head(h, z)
        return {
            "h": h,
            "z": z,
            "z_prior": z_prior,
            "post_dist": post_dist,
            "prior_dist": prior_dist,
            "reconstructed": recon,
            "reward": pred_reward,
            "value": pred_value,
            "discount": pred_discount
        }
