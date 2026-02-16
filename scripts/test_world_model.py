import torch
from dreamer4.models.encoder import Encoder
from dreamer4.models.rssm import RSSM
from dreamer4.models.decoder import Decoder
from dreamer4.models.heads import RewardHead, ValueHead

def main():
    # ---- Configuration ----
    config = {
        "env": {
            "channels": 3,
            "image_size": 64,
            "action_dim": 6,
        },
        "model": {
            "embedding_dim": 1024,
            "hidden_dim": 200,
            "latent_dim": 32,
            "categories": 32,
            "encoder_channels": [32, 64, 128, 256],
            "decoder_channels": [256, 128, 64, 32],  # example decoder channels
        }
    }

    batch_size = 8

    # ---- Dummy input ----
    images = torch.randn(batch_size, 3, 64, 64)
    prev_a = torch.randn(batch_size, config["env"]["action_dim"])
    prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"])
    prev_z = torch.zeros(
        batch_size,
        config["model"]["latent_dim"],
        config["model"]["categories"],
    )
    prev_z[:, :, 0] = 1  # initial one-hot latent

    # ---- Encoder ----
    encoder = Encoder(config)
    embed = encoder(images)
    print("Embedding shape:", embed.shape)  # (B, embedding_dim)

    # ---- RSSM ----
    rssm = RSSM(
        config["env"]["action_dim"],
        config["model"]["embedding_dim"],
        config["model"]["hidden_dim"],
        config["model"]["latent_dim"],
        config["model"]["categories"],
    )

    # ---- Discrete sampling ----
    h, z, z_prior, post_dist, prior_dist = rssm(
        prev_h, prev_z, prev_a, embed, use_relaxed=False
    )
    print("\nDiscrete sampling shapes:")
    print("h:", h.shape)
    print("z:", z.shape)
    print("z_prior:", z_prior.shape)

    # ---- Decoder ----
    decoder = Decoder(config)
    reconstructed_images = decoder(h, z)
    print("\nReconstructed image shape:", reconstructed_images.shape)  # (B, C, H, W)

    # ---- Reward and Value Heads ----
    reward_head = RewardHead(
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        categories=config["model"]["categories"]
    )
    value_head = ValueHead(
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        categories=config["model"]["categories"]
    )

    pred_reward = reward_head(h, z)
    pred_value = value_head(h, z)
    print("\nReward head output shape:", pred_reward.shape)  # (B,1)
    print("Value head output shape:", pred_value.shape)      # (B,1)

    # ---- Relaxed sampling + gradient check ----
    h_relax, z_relax, z_prior_relax, _, _ = rssm(
        prev_h, prev_z, prev_a, embed, use_relaxed=True, temperature=0.67
    )

    reconstructed_images_relax = decoder(h_relax, z_relax)
    pred_reward_relax = reward_head(h_relax, z_relax)
    pred_value_relax = value_head(h_relax, z_relax)

    # ---- Gradient check ----
    (reconstructed_images_relax.mean() +
     pred_reward_relax.mean() +
     pred_value_relax.mean()).backward()
    print("\nGradients flow correctly through full model (encoder, RSSM, decoder, heads)")

if __name__ == "__main__":
    main()
