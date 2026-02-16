import torch
import torch.optim as optim
import yaml 
from dreamer4.models.world_model import WorldModel
from dreamer4.training.losses import world_model_loss

def mini_train_step(config, batch_size=8, seq_len=5, lr=1e-3):
    """
    Runs a mini training step on dummy data to test WorldModel + losses.
    """
    # ---- Dummy trajectory data ----
    obs = torch.randn(batch_size, seq_len, config["env"]["channels"],
                      config["env"]["image_size"], config["env"]["image_size"])
    actions = torch.randn(batch_size, seq_len, config["env"]["action_dim"])
    rewards = torch.randn(batch_size, seq_len, 1)

    # ---- Initialize WorldModel and optimizer ----
    wm = WorldModel(config)
    optimizer = optim.Adam(wm.parameters(), lr=lr)

    # ---- Initialize hidden state and latent ----
    prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"])
    prev_z = torch.zeros(batch_size, config["model"]["latent_dim"], config["model"]["categories"])
    prev_z[:, :, 0] = 1  # one-hot init

    total_loss = 0.0

    # ---- Loop over trajectory ----
    for t in range(seq_len):
        obs_t = obs[:, t]
        act_t = actions[:, t]
        reward_t = rewards[:, t]

        # ---- Forward pass ----
        out = wm(obs_t, act_t, prev_h, prev_z, use_relaxed=True, temperature=0.67)

        # ---- Compute losses ----
        step_loss, obs_loss, reward_loss, kl_loss = world_model_loss(out, obs_t, reward_t)
        total_loss += step_loss

        # ---- Update prev_h and prev_z for next step ----
        prev_h = out["h"].detach()
        prev_z = out["z"].detach()

    # ---- Backprop ----
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("Mini training step complete!")
    print(f"Total loss: {total_loss.item():.4f}")

# ---- Run test if called directly ----
if __name__ == "__main__":
    # Load YAML config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mini_train_step(config)
