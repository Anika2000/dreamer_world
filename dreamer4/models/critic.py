import torch
import torch.nn as nn
#The actor alone could learn by trial and error — but that’s often slow because it doesn’t know how good its actions are yet.
#Critic = value network → predicts how good the current state is (expected future rewards).
# Why it helps:
# Instead of waiting until the end of a trajectory to see the total reward, the critic estimates it immediately.
# This guides the actor: the actor can improve by moving towards actions that lead to higher predicted value.
class Critic(nn.Module):
    #Input: latent state (h_t, z_t)
    #Output: value V(s_t) (scalar)
    def __init__(self, hidden_dim, latent_dim, categories):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim * categories, 400),
            nn.ReLU(),
            nn.Linear(400, 1) #final layer outputs a single number, the estimated value V(s_t) of this latent state.
        )

    def forward(self, h, z):
        z = z.view(z.size(0), -1)
        x = torch.cat([h, z], dim=-1)
        value = self.fc(x)
        return value
