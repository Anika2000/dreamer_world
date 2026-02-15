import torch

from dreamer4.models.encoder import Encoder
from dreamer4.models.rssm import RSSM
def main():
    # ---- Configuration ----
    batch_size = 8
    channels = 3
    width, height = 64, 64
    embedding_dim = 1024
    action_dim = 6   # example: 6-DoF for robot arm
    hidden_dim = 200
    latent_dim = 32
    categories = 32

    # ---- Dummy input ----
    # Simulate a batch of images
    images = torch.randn(batch_size, channels, width, height)
    # Simulate previous action and previous latent
    prev_a = torch.randn(batch_size, action_dim)
    prev_h = torch.zeros(batch_size, hidden_dim)
    prev_z = torch.zeros(batch_size, latent_dim, categories)
    prev_z[:, :, 0] = 1  # simple one-hot initial z

    # ---- Encoder ----
    encoder = Encoder(channels=channels, embedding_dim=embedding_dim)
    embed = encoder(images)
    print("Embedding shape:", embed.shape)  # should be (B, embedding_dim)

    # ---- RSSM ----
    rssm = RSSM(action_dim, embedding_dim, hidden_dim, latent_dim, categories)

    # ---- Test discrete sampling ----
    h, z, z_prior, post_dist, prior_dist = rssm(prev_h, prev_z, prev_a, embed, use_relaxed=False)
    print("Discrete sampling shapes:")
    print("h:", h.shape)
    print("z:", z.shape)
    print("z_prior:", z_prior.shape)

    # ---- Test relaxed sampling (training-ready) ----
    h_relax, z_relax, z_prior_relax, post_dist_relax, prior_dist_relax = rssm(
        prev_h, prev_z, prev_a, embed, use_relaxed=True, temperature=0.67
    )
    print("\nRelaxed sampling shapes (differentiable):")
    print("h:", h_relax.shape)
    print("z:", z_relax.shape)
    print("z_prior:", z_prior_relax.shape)

    # ---- Optional gradient check ----
    z_relax.sum().backward()  # should propagate gradients without errors
    print("\nGradients flow correctly for relaxed z")

if __name__ == "__main__":
    main()
