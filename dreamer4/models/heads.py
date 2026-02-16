import torch
import torch.nn as nn

#RewardHead → predicts the reward you would get from a given latent state.
class RewardHead(nn.Module):
    #Input: h_t (deterministic RSSM hidden state) + z_t (stochastic latent)
    #Process: flatten z_t, concatenate with h_t, pass through a small MLP
    #Output: scalar reward r_t
    def __init__(self, hidden_dim, latent_dim, categories):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim * categories, 400),
            nn.ReLU(),
            nn.Linear(400, 1)  # scalar reward
        )

    def forward(self, h, z):
        z = z.view(z.size(0), -1)        # flatten latent
        x = torch.cat([h, z], dim=-1)    # concat h + z
        reward = self.fc(x) #this is the function that is the MLP inside the __init__ 
        return reward

#ValueHead → predicts the value (expected future return) for that latent state 
# — used in actor-critic / imagination rollouts.
class ValueHead(nn.Module):
    #Input: same as above
    #Process: same small MLP
    #Output: scalar value estimate V(s_t)
    def __init__(self, hidden_dim, latent_dim, categories):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim * categories, 400),
            nn.ReLU(),
            nn.Linear(400, 1)  # scalar value
        )

    def forward(self, h, z):
        z = z.view(z.size(0), -1)
        x = torch.cat([h, z], dim=-1)
        value = self.fc(x)
        return value
