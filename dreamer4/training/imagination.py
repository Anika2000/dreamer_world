import torch

#This function is simulating trajectories in the latent space, not in the real environment.
def imagination_rollout(rssm, actor, start_h, start_z, wm, horizon=15):
    #Input: 
    #rssm: predicts next latent states (h and z) given the current latent state and an action.
    """
    Rollout imagined trajectories in latent space using current actor.
    Returns imagined hidden states, latent states, actions, predicted rewards, and discounts.
    """
    h, z = start_h, start_z
    imagined_h, imagined_z, actions, rewards, discounts = [], [], [], [], []

    for t in range(horizon):
        #Take the current latent state (h, z).
        a = actor(h, z)                 # action from policy
        #Feed it into rssm along with the action from the current actor.
        #Get the next latent state (h_next, z_next).
        h, z, _, _, _ = rssm(h, z, a, embed=None, use_relaxed=True)  # imagined next latent
        imagined_h.append(h)
        imagined_z.append(z)
        actions.append(a)
        r = wm.reward_head(h, z)
        d = wm.discount_head(h, z)
        rewards.append(r)
        discounts.append(d)

    # Stack tensors along time dimension: (T, B, hidden_dim)
    #T = time steps / horizon of imagination
    #B = batch size (number of parallel imagined trajectories)
    #hidden_dim = dimension of the deterministic hidden state h
    #Shape of imagined_h and imagined_z: (T, B, hidden_dim/latent_dim)
    imagined_h = torch.stack(imagined_h, dim=0)
    imagined_z = torch.stack(imagined_z, dim=0)
    #Shape of actions: (T, B, action_dim)
    actions = torch.stack(actions, dim=0)

    return imagined_h, imagined_z, actions, rewards, discounts

def compute_actor_critic_loss(value_head, imagined_h, imagined_z, pred_rewards, pred_discounts, gamma=0.99, lam=0.95):
    #Inputs: 
    #value_head predicts the expected return for each latent state.
    #imagined_h and imagined_z are the imagined trajectories.
    """
    Compute Dreamer V2-style actor and critic losses along imagined trajectories.
    
    Args:
        value_head: the value network predicting expected returns.
        imagined_h, imagined_z: tensors of shape (T, B, hidden_dim/latent_dim).
        pred_rewards: predicted rewards along imagined trajectory, shape (T, B)
        pred_discounts: predicted discounts along imagined trajectory, shape (T, B)
        gamma: discount factor
        lam: lambda for TD(lambda)
        entropy_coef: optional coefficient for actor entropy regularization
        imagined_a: actions along imagined trajectory, used for entropy regularization
    
    Returns:
        actor_loss, critic_loss (scalars)
    """
    horizon, batch_size, _ = imagined_h.shape

    #Every PyTorch tensor lives on a device, e.g., CPU or GPU.
    #.device gets that device (cpu or cuda:0).
    #This line ensures that any new tensors you create (like returns or G) 
    # are on the same device as imagined_h, avoiding runtime errors.
    #Without this, if you try to do math between a CPU tensor and a GPU tensor, PyTorch will throw an error.
    device = imagined_h.device

    # Compute predicted values at each step along the imagined trajectory.
    #Combines the immediate predicted reward (here zero for simplicity) and bootstrapped value predictions.
    #value_head(h, z) predicts the value of each imagined latent state
    #Think of this as: “If I start at imagined step t, this is how good I think I’ll do from here on.”
    values = torch.stack([value_head(h, z) for h, z in zip(imagined_h, imagined_z)], dim=0)
    values = values.squeeze(-1)  # (T, B)

    # Compute lambda-returns
    #returns will store the lambda-returns for every imagined step.
    returns = torch.zeros_like(values)
    #G is a running accumulator: think of it as “the return from the future we’ll combine with the current predicted value.”
    G = torch.zeros(batch_size, device=device)

    #This is the lambda-return formula (TD(λ)):
    #1. r = immediate reward (here zero because in your simplified dummy rollout we’re not using reward predictions yet)
    #2. values[t] = predicted value of current latent state
    #3. G = previous future return
    #gamma = discount factor
    #λ (lambda) = controls bias-variance tradeoff:λ = 0 → purely 1-step TD target; λ = 1 → full Monte Carlo return (sum of all future rewards)
    #We loop backwards because each step depends on the next step’s return.
    for t in reversed(range(horizon)):
        r = torch.zeros(batch_size, device=device)  # reward head not used here, could add if desired
        G = r + gamma * ((1 - lam) * values[t] + lam * G)
        returns[t] = G

    # Critic loss = MSE of value predictions vs returns
    critic_loss = ((values - returns)**2).mean()

    # Actor loss = maximize imagined returns (gradient ascent)
    actor_loss = -returns.mean()

    return actor_loss, critic_loss
