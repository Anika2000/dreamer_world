#first: World Model Training → learns latent dynamics (h, z) from real environment.
#second:  Imagination Rollout → generates sequences in latent space using current actor.
#third: updates policy and value function using imagined trajectories.
import torch
import torch.optim as optim
import numpy as np
import yaml

from dreamer4.models.world_model import WorldModel
from dreamer4.models.actor import Actor
from dreamer4.models.critic import Critic
from dreamer4.training.losses import world_model_loss
from dreamer4.training.imagination import imagination_rollout, compute_actor_critic_loss
from dreamer4.envs.so101_env import SO101Env
from dreamer4.training.replay_buffer import ReplayBuffer


def collect_trajectories(env, actor, buffer, seq_len=5, num_sequences=10, device="cpu"):
    actor.eval()
    for seq_idx in range(num_sequences):
        obs_seq, action_seq, reward_seq = [], [], []
        obs, _ = env.reset()

        for t in range(seq_len):
            obs_tensor = torch.tensor(obs.transpose(2,0,1), dtype=torch.float32, device=device).unsqueeze(0)/255.0
            with torch.no_grad():
                # dummy previous states
                prev_h = torch.zeros(1, actor.fc[0].in_features - env.action_space.shape[0], device=device)
                prev_z = torch.zeros(1, actor.fc[0].in_features - prev_h.size(1), 1, device=device)
                action = actor(prev_h, prev_z).cpu().numpy()[0]

            next_obs, reward, done, truncated, _ = env.step(action)

            obs_seq.append(obs)
            action_seq.append(action)
            reward_seq.append([reward])

            obs = next_obs
            if done:
                obs, _ = env.reset()

        buffer.add_sequence(
            np.array(obs_seq, dtype=np.uint8).transpose(0,3,1,2),
            np.array(action_seq, dtype=np.float32),
            np.array(reward_seq, dtype=np.float32)
        )
    print(f"Collected {num_sequences} sequences. Buffer size: {len(buffer)}")


def full_train_step(config, steps=50, batch_size=2, seq_len=5, buffer_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init environment
    env = SO101Env(image_size=config["env"]["image_size"])

    # Networks
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

    # Replay buffer
    buffer = ReplayBuffer(
        max_size=buffer_size,
        seq_len=seq_len,
        obs_shape=(config["env"]["channels"], config["env"]["image_size"], config["env"]["image_size"]),
        action_dim=config["env"]["action_dim"]
    )

    # -----------------------------
    # Collect initial data
    # -----------------------------
    collect_trajectories(env, actor, buffer, seq_len=seq_len, num_sequences=10, device=device)

    # -----------------------------
    # Main training loop
    # -----------------------------
    for step in range(steps):
        if buffer.size < batch_size:
            continue

        obs_batch, action_batch, reward_batch = buffer.sample(batch_size)
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)/255.0
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)

        # dummy init for hidden/latent
        prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"], device=device)
        prev_z = torch.zeros(batch_size, config["model"]["latent_dim"], config["model"]["categories"], device=device)
        prev_z[:, :, 0] = 1

        # WORLD MODEL UPDATE
        wm_loss_total = 0
        for t in range(seq_len):
            obs_t = obs_batch[:, t]
            act_t = action_batch[:, t]
            reward_t = reward_batch[:, t]

            out = wm(obs_t, act_t, prev_h, prev_z, use_relaxed=True)
            wm_loss_tuple = world_model_loss(out, obs_t, reward_t)
            wm_loss = wm_loss_tuple[0] if isinstance(wm_loss_tuple, tuple) else wm_loss_tuple
            wm_loss_total += wm_loss
            prev_h = out["h"].detach()
            prev_z = out["z"].detach()

        wm_opt.zero_grad()
        wm_loss_total.backward()
        wm_opt.step()

        # IMAGINATION + ACTOR/CRITIC
        imagined_h, imagined_z, _ = imagination_rollout(wm.rssm, actor, prev_h, prev_z, horizon=10)
        actor_loss, critic_loss = compute_actor_critic_loss(wm.value_head, imagined_h, imagined_z)

        actor_opt.zero_grad()
        critic_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        actor_opt.step()
        critic_opt.step()

        if step % 5 == 0:
            print(f"Step {step} | WM Loss: {wm_loss_total.item():.4f} | "
                  f"Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | Buffer: {buffer.size}")


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    full_train_step(config)