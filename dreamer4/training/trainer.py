import torch
import torch.optim as optim
import yaml

from dreamer4.models.world_model import WorldModel
from dreamer4.models.actor import Actor
from dreamer4.models.critic import Critic
from dreamer4.training.losses import world_model_loss
from dreamer4.training.imagination import imagination_rollout, compute_actor_critic_loss

def full_train_step(config, steps=50, batch_size=8, seq_len=5, imagination_horizon=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    wm = WorldModel(config).to(device)
    actor = Actor(
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        categories=config["model"]["categories"],
        action_dim=config["env"]["action_dim"]
    ).to(device)
    critic = Critic(
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        categories=config["model"]["categories"]
    ).to(device)

    # Optimizers
    wm_opt = optim.Adam(wm.parameters(), lr=float(config["training"]["lr"]))
    actor_opt = optim.Adam(actor.parameters(), lr=float(config["training"]["lr"]))
    critic_opt = optim.Adam(critic.parameters(), lr=float(config["training"]["lr"]))

    for step in range(steps):
        # --- Sample dummy data (replace with real environment later) ---
        obs = torch.randn(batch_size, seq_len, config["env"]["channels"],
                          config["env"]["image_size"], config["env"]["image_size"]).to(device)
        actions = torch.randn(batch_size, seq_len, config["env"]["action_dim"]).to(device)
        rewards = torch.randn(batch_size, seq_len, 1).to(device)

        prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"], device=device)
        prev_z = torch.zeros(batch_size, config["model"]["latent_dim"], config["model"]["categories"], device=device)
        prev_z[:, :, 0] = 1  # one-hot initialization

        # --- WORLD MODEL UPDATE ---
        wm_loss_total = 0
        for t in range(seq_len):
            obs_t = obs[:, t]
            act_t = actions[:, t]
            reward_t = rewards[:, t]

            out = wm(obs_t, act_t, prev_h, prev_z, use_relaxed=True)

            wm_loss, obs_loss, reward_loss, kl_loss, discount_loss = world_model_loss(
                out, obs_t, reward_t,
                discount=torch.ones(batch_size, 1, device=device) * 0.999,
                beta=float(config["training"]["kl_scale"])
            )
            wm_loss_total += wm_loss

            prev_h = out["h"].detach()
            prev_z = out["z"].detach()

        wm_opt.zero_grad()
        wm_loss_total.backward()
        wm_opt.step()

        # --- IMAGINATION ROLLOUT + ACTOR/CRITIC UPDATE ---
        imagined_h, imagined_z, _ = imagination_rollout(wm.rssm, actor, prev_h, prev_z, horizon=imagination_horizon)

        actor_loss, critic_loss = compute_actor_critic_loss(wm.value_head, imagined_h, imagined_z)

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        if step % 5 == 0:
            print(f"Step {step}, WM Loss: {wm_loss_total.item():.4f}, "
                  f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")


if __name__ == "__main__":
    # Load YAML config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    full_train_step(config)
