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
from dreamer4.training.imagination import imagination_rollout
from dreamer4.envs.so101_env import SO101Env
from dreamer4.training.replay_buffer import ReplayBuffer
from dreamer4.utils.model_logger import ModelLogger


def collect_trajectories(env, actor, buffer, seq_len=5, num_sequences=50, device="cpu"):
    actor.eval()
    for seq_idx in range(num_sequences):
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        obs, _ = env.reset()

        # Initialize hidden and latent states
        prev_h = torch.zeros(1, config["model"]["hidden_dim"], device=device)
        prev_z = torch.zeros(1, config["model"]["latent_dim"], config["model"]["categories"], device=device)
        prev_z[:, :, 0] = 1

        for t in range(seq_len):
            # DEBUG: check if frames change
            if t == 0:
                first = obs.copy()


            # The reason we sometimes take random actions instead of always 
            # using the actor is exploration:

            # Early in training, the actor is untrained and its outputs are basically random. 
            # If we only follow the actor, we might only see a very narrow set of states, 
            # which prevents the world model from learning the full dynamics of the environment.

            # Random actions help cover more of the state space, so the world 
            # model sees more varied transitions — like the arm moving to different positions, 
            # grabbing different objects, or failing. This improves the quality of latent dynamics learned.

            # Later, as the actor improves, you can gradually reduce random actions 
            # and rely more on the actor for trajectory collection.
            if np.random.rand() < 0.5:
                action = env.action_space.sample()  # random
            else:
                obs_tensor = torch.tensor(obs.transpose(2,0,1), dtype=torch.float32, device=device).unsqueeze(0)/255.0
                with torch.no_grad():
                    action, _ = actor(prev_h, prev_z)
                    action = action.cpu().numpy()[0]

            next_obs, reward, done, truncated, _ = env.step(action)

            obs_seq.append(obs)
            action_seq.append(action)
            reward_seq.append([reward])
            done_seq.append([done])
            obs = next_obs
            # DEBUG: compare first and last frame
            if t == seq_len - 1:
                print("frame diff:", np.mean(np.abs(first - obs)))
            if done:
                obs, _ = env.reset()

        buffer.add_sequence(
            np.array(obs_seq, dtype=np.uint8).transpose(0, 3, 1, 2),
            np.array(action_seq, dtype=np.float32),
            np.array(reward_seq, dtype=np.float32),
            np.array(done_seq, dtype=np.float32)
        )
    print(f"Collected {num_sequences} sequences. Buffer size: {len(buffer)}")


#we increase the steps from 50 to 5000 so new sequences are collected from actor policy — helps generalization.
def full_train_step(config, steps=250, batch_size=16, seq_len=5, buffer_size=1000, tau=0.99,
                    kl_beta=0.1, entropy_coef=1e-2, device="cuda"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the logger
    logger = ModelLogger(log_dir="logs")
    #Environment 
    env = SO101Env(image_size=config["env"]["image_size"])

    #Models
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
    target_critic = Critic(
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        categories=config["model"]["categories"]
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())

    #Optimizers
    wm_opt = optim.Adam(wm.parameters(), lr=float(config["training"]["lr"]))
    actor_opt = optim.Adam(actor.parameters(), lr=float(config["training"]["lr"]))
    critic_opt = optim.Adam(critic.parameters(), lr=float(config["training"]["lr"]))

    #Replay Buffer
    buffer = ReplayBuffer(
        max_size=buffer_size,
        seq_len=seq_len,
        obs_shape=(config["env"]["channels"], config["env"]["image_size"], config["env"]["image_size"]),
        action_dim=config["env"]["action_dim"]
    )

    # Collect initial trajectories
    collect_trajectories(env, actor, buffer, seq_len=seq_len, num_sequences=batch_size, device=device)

    #Training loop
    for step in range(steps):
        if len(buffer) < batch_size:
            continue
        
        # Sample a batch
        obs_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

        #print("obs_batch min/max before scaling:", obs_batch.min(), obs_batch.max())

        prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"], device=device)
        prev_z = torch.zeros(batch_size, config["model"]["latent_dim"], config["model"]["categories"], device=device)
        prev_z[:, :, 0] = 1

        # -----------------------------
        # WORLD MODEL UPDATE
        # -----------------------------
        wm_loss_total = 0
        img_loss_total = 0
        rew_loss_total = 0
        kl_loss_total = 0
        disc_loss_total = 0
        for t in range(seq_len):
            obs_t = obs_batch[:, t]
            act_t = action_batch[:, t]
            reward_t = reward_batch[:, t]
            discount_t = buffer.compute_discounts(done_batch[:, t], gamma=0.99)
            # Log input observation at this step
            if t == 0 and step % 50 == 0:  # Log every 50 steps
                logger.log_image(obs_batch[0, t], step) 
            out = wm(obs_t, act_t, prev_h, prev_z, use_relaxed=True)
            loss, image_loss, reward_loss, kl_loss, discount_loss = world_model_loss(out, obs_t, reward_t, discount_t, beta=kl_beta, alpha=0.8)
            wm_loss_total += loss
            img_loss_total += image_loss
            rew_loss_total += reward_loss
            kl_loss_total += kl_loss
            disc_loss_total += discount_loss
            prev_h = out["h"]
            prev_z = out["z"]
        wm_loss_total = wm_loss_total / seq_len
        img_loss_total /= seq_len
        rew_loss_total /= seq_len
        kl_loss_total /= seq_len
        disc_loss_total /= seq_len
        wm_opt.zero_grad()
        wm_loss_total.backward() # Only use wm_loss_total
        wm_opt.step()

        # -----------------------------
        # IMAGINATION + ACTOR/CRITIC (Dreamer V2 style)
        # -----------------------------
        h_actor, z_actor = prev_h.detach(), prev_z.detach()  # stop gradients from world model
        # Temporarily freeze world model during imagination
        for p in wm.parameters():
            p.requires_grad = False

        imagined_h, imagined_z, imagined_a, imagined_log_probs, pred_rewards, pred_discounts = imagination_rollout(
            wm.rssm, actor, h_actor, z_actor, wm=wm, horizon=10
        )
        # Re-enable world model gradients
        for p in wm.parameters():
            p.requires_grad = True

        # Convert predicted rewards and discounts to tensors
        pred_rewards = torch.stack(pred_rewards, dim=0).squeeze(-1)
        pred_discounts = torch.stack(pred_discounts, dim=0).squeeze(-1)

        # Compute predicted values for imagined states
        values = torch.stack(
            [critic(h.detach(), z.detach()) for h, z in zip(imagined_h, imagined_z)], dim=0
        ).squeeze(-1)  # shape: (T, B)
        
        with torch.no_grad():
            target_values = torch.stack(
                [target_critic(h.detach(), z.detach()) for h, z in zip(imagined_h, imagined_z)],
                dim=0
            ).squeeze(-1)

        # Lambda-return computation (TD(lambda)) with predicted rewards & discounts
        horizon, batch_size = values.shape
        returns = torch.zeros_like(values)
        G = torch.zeros(batch_size, device=values.device)
        gamma = 0.99
        lam = 0.95
        for t in reversed(range(horizon)):
            G = pred_rewards[t] + pred_discounts[t] * ((1 - lam) * target_values[t] + lam * G)
            returns[t] = G

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Critic loss: MSE between value predictions and lambda-returns
        critic_loss = ((values - returns)**2).mean()
        # Actor loss: maximize imagined returns
        advantage = returns - values.detach()
        actor_loss = -(advantage * imagined_log_probs.squeeze(-1)).mean()

        # Optimize actor and critic
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        actor_opt.step()
        critic_opt.step()

        # EMA update
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * param.data

        if step % 5 == 0:
            print(
                f"Step {step} | "
                f"WM: {wm_loss_total.item():.4f} | "
                f"Img: {img_loss_total.item():.4f} | "
                f"Rew: {rew_loss_total.item():.4f} | "
                f"KL: {kl_loss_total.item():.4f} | "
                f"Disc: {disc_loss_total.item():.4f} | "
                f"Actor: {actor_loss.item():.4f} | "
                f"Critic: {critic_loss.item():.4f}"
            )
        if step % 500 == 0 and step > 0:
            collect_trajectories(env, actor, buffer, seq_len=seq_len, num_sequences=50, device=device)


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    full_train_step(config)